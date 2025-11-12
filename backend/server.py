import os
import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ========== Konfigurasi dasar ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ========== Inisialisasi Aplikasi ==========
app = FastAPI(title="News Classifier API")

# ========== CORS ==========
origins = os.getenv("CORS_ORIGIN", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Koneksi ke MongoDB ==========
mongo_url = os.getenv("MONGO_URL")
db_name = os.getenv("DB_NAME", "klasifikasi_berita")
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# ========== Preprocessing tools ==========
stemmer_factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_remover = stopword_factory.create_stop_word_remover()

# ========== Variabel Model ==========
vectorizer = None
knn_classifier = None
categories = ["Ekonomi", "Hiburan", "Olahraga", "Politik", "Teknologi"]

# ========== Pydantic Models ==========
class News(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    judul: str
    isi: str
    kategori: str
    confidence_score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NewsCreate(BaseModel):
    judul: str
    isi: str

class CategoryStats(BaseModel):
    kategori: str
    jumlah: int

# ========== Helper Functions ==========
def simplify_category(cat: str) -> str:
    cat = str(cat)
    for key in categories:
        if key in cat:
            return key
    return "Lainnya"

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# ========== Model Training & Prediction ==========
async def train_model():
    global vectorizer, knn_classifier
    logging.info("Memulai pelatihan model...")

    all_news = await db.news.find({}, {"_id": 0}).to_list(10000)
    if len(all_news) < 5:
        logging.warning("Data kurang dari 5, model mungkin tidak akurat.")
        if len(all_news) == 0:
            logging.error("Tidak ada data. Model tidak bisa dilatih.")
            return

    texts = []
    labels = []
    for item in all_news:
        combined = preprocess_text(item["judul"] + " " + item["isi"])
        texts.append(combined)
        labels.append(item["kategori"])

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    n_neighbors = min(5, len(texts))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn_classifier.fit(X, labels)
    logging.info(f"Model berhasil dilatih dengan {len(texts)} data.")

def classify_news(judul: str, isi: str):
    if vectorizer is None or knn_classifier is None:
        logging.warning("Model belum dilatih, klasifikasi dibatalkan.")
        return "Unknown", 0.0

    processed = preprocess_text(judul + " " + isi)
    X_new = vectorizer.transform([processed])
    probs = knn_classifier.predict_proba(X_new)[0]
    idx = np.argmax(probs)
    return knn_classifier.classes_[idx], float(probs[idx])

# ========== FastAPI Startup & Shutdown ==========
@app.on_event("startup")
async def startup_event():
    logging.info("Server startup...")
    count = await db.news.count_documents({})

    if count == 0:
        logging.warning("Database kosong, memuat data awal dari CSV...")
        try:
            df = pd.read_csv("cnnindonesia_scraped.csv")
            df["judul"] = df["judul"].fillna("")
            df["isi"] = df["isi"].fillna("")
            df["kategori"] = df["kategori"].apply(simplify_category)
            df = df[df["kategori"].isin(categories)]

            initial_docs = []
            for cat in categories:
                sample_count = min(10, len(df[df["kategori"] == cat]))
                if sample_count > 0:
                    samples = df[df["kategori"] == cat].sample(n=sample_count, random_state=42)
                    for _, row in samples.iterrows():
                        news = News(judul=row["judul"], isi=row["isi"], kategori=row["kategori"], confidence_score=1.0)
                        doc = news.model_dump()
                        doc["created_at"] = doc["created_at"].isoformat()
                        initial_docs.append(doc)
            if initial_docs:
                await db.news.insert_many(initial_docs)
                logging.info(f"Berhasil menambahkan {len(initial_docs)} data awal.")
        except Exception as e:
            logging.error(f"Gagal memuat CSV: {e}")
    else:
        logging.info(f"Database sudah berisi {count} data.")

    # Jalankan pelatihan model di background thread agar port cepat terbuka
    def background_training():
        import asyncio
        asyncio.run(train_model())
        logging.info("Pelatihan model selesai (background).")

    threading.Thread(target=background_training, daemon=True).start()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    logging.info("Koneksi MongoDB ditutup.")

# ========== API ROUTES ==========
router = APIRouter(prefix="/api")

@router.get("/")
async def root():
    return {"message": "News Classifier API aktif!"}

@router.post("/news", response_model=News)
async def create_news(input: NewsCreate):
    kategori, confidence = classify_news(input.judul, input.isi)
    news_obj = News(judul=input.judul, isi=input.isi, kategori=kategori, confidence_score=confidence)
    doc = news_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.news.insert_one(doc)
    threading.Thread(target=lambda: asyncio.run(train_model()), daemon=True).start()
    return news_obj

@router.get("/news", response_model=List[News])
async def get_all_news():
    news_list = await db.news.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for item in news_list:
        if isinstance(item["created_at"], str):
            item["created_at"] = datetime.fromisoformat(item["created_at"])
    return news_list

@router.get("/news/{kategori}", response_model=List[News])
async def get_news_by_category(kategori: str):
    if kategori not in categories:
        raise HTTPException(status_code=404, detail="Kategori tidak ditemukan")
    news_list = await db.news.find({"kategori": kategori}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for item in news_list:
        if isinstance(item["created_at"], str):
            item["created_at"] = datetime.fromisoformat(item["created_at"])
    return news_list

@router.get("/categories/stats", response_model=List[CategoryStats])
async def get_category_stats():
    stats = []
    for cat in categories:
        count = await db.news.count_documents({"kategori": cat})
        stats.append(CategoryStats(kategori=cat, jumlah=count))
    return stats

@router.post("/train")
async def retrain_model_manual():
    threading.Thread(target=lambda: asyncio.run(train_model()), daemon=True).start()
    return {"message": "Pelatihan model dimulai di background"}

# ========== Register Router ==========
app.include_router(router)

# ========== Main Entry Point ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
