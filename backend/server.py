from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import asyncio

# === Machine Learning Libraries ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# === Konfigurasi dasar ===
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# === FastAPI App ===
app = FastAPI(title="News Classifier API")

# === Konfigurasi CORS ===
origins = os.getenv("CORS_ORIGIN", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Koneksi MongoDB ===
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# === Router ===
api_router = APIRouter(prefix="/api")

# === Preprocessing tools ===
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# === Model global ===
vectorizer = None
knn_classifier = None
categories = ["Ekonomi", "Hiburan", "Olahraga", "Politik", "Teknologi"]

# === Data Models ===
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


# === Helper Functions ===
def simplify_category(cat: str) -> str:
    cat = str(cat)
    if "Politik" in cat:
        return "Politik"
    elif "Olahraga" in cat:
        return "Olahraga"
    elif "Ekonomi" in cat:
        return "Ekonomi"
    elif "Hiburan" in cat:
        return "Hiburan"
    elif "Teknologi" in cat:
        return "Teknologi"
    else:
        return "Lainnya"

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


# === Model Training ===
async def train_model():
    global vectorizer, knn_classifier

    logging.info("Memulai pelatihan model...")
    all_news = await db.news.find({}, {"_id": 0}).to_list(10000)

    if len(all_news) == 0:
        logging.warning("Tidak ada data untuk melatih model.")
        return

    texts = []
    labels = []
    for news_item in all_news:
        combined_text = news_item["judul"] + " " + news_item["isi"]
        processed_text = preprocess_text(combined_text)
        texts.append(processed_text)
        labels.append(news_item["kategori"])

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    n_neighbors = min(5, len(texts))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn_classifier.fit(X, labels)

    logging.info(f"Model berhasil dilatih dengan {len(texts)} data.")


def classify_news(judul: str, isi: str) -> tuple:
    if vectorizer is None or knn_classifier is None:
        logging.warning("Model belum dilatih.")
        return "Unknown", 0.0

    combined_text = judul + " " + isi
    processed_text = preprocess_text(combined_text)
    X_new = vectorizer.transform([processed_text])
    probabilities = knn_classifier.predict_proba(X_new)[0]

    max_prob_index = np.argmax(probabilities)
    predicted_category = knn_classifier.classes_[max_prob_index]
    confidence = float(np.max(probabilities))

    return predicted_category, confidence


# === Startup Event ===
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
                samples = df[df["kategori"] == cat].sample(n=min(10, len(df[df["kategori"] == cat])), random_state=42)
                for _, row in samples.iterrows():
                    news = News(
                        judul=row["judul"],
                        isi=row["isi"],
                        kategori=row["kategori"],
                        confidence_score=1.0
                    )
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

    # ✅ Jalankan training di loop yang sama, bukan di thread lain
    async def async_background_training():
        try:
            await train_model()
            logging.info("Pelatihan model selesai (background task).")
        except Exception as e:
            logging.error(f"Error di background training: {e}")

    asyncio.create_task(async_background_training())


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    logging.info("Koneksi MongoDB ditutup.")


# === API Endpoints ===
@api_router.get("/")
async def root():
    return {"message": "News Classifier API"}

@api_router.post("/news", response_model=News)
async def create_news(input: NewsCreate):
    kategori, confidence = classify_news(input.judul, input.isi)
    news_obj = News(judul=input.judul, isi=input.isi, kategori=kategori, confidence_score=confidence)
    doc = news_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.news.insert_one(doc)

    # Latih ulang secara async agar tidak blokir request
    asyncio.create_task(train_model())

    return news_obj

@api_router.get("/news", response_model=List[News])
async def get_all_news():
    news_list = await db.news.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for news_item in news_list:
        if isinstance(news_item["created_at"], str):
            news_item["created_at"] = datetime.fromisoformat(news_item["created_at"])
    return news_list

@api_router.get("/news/{kategori}", response_model=List[News])
async def get_news_by_category(kategori: str):
    if kategori not in categories:
        raise HTTPException(status_code=404, detail="Kategori tidak ditemukan")
    news_list = await db.news.find({"kategori": kategori}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for news_item in news_list:
        if isinstance(news_item["created_at"], str):
            news_item["created_at"] = datetime.fromisoformat(news_item["created_at"])
    return news_list

@api_router.get("/categories/stats", response_model=List[CategoryStats])
async def get_category_stats():
    stats = []
    for kategori in categories:
        count = await db.news.count_documents({"kategori": kategori})
        stats.append(CategoryStats(kategori=kategori, jumlah=count))
    return stats

@api_router.post("/train")
async def retrain_model_manual():
    asyncio.create_task(train_model())
    return {"message": "Model retrained in background"}

@app.get("/", tags=["Health"])
async def health_check():
    """
    Endpoint dasar untuk memastikan server hidup.
    Render menggunakan ini untuk health check otomatis.
    """
    return {"status": "ok", "message": "Backend News Classifier API is running!"}

# === Router ===
app.include_router(api_router)

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# === Entry Point ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
