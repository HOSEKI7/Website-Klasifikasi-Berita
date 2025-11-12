from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import pandas as pd  # Pastikan ini ada di requirements.txt
import numpy as np

# Import library ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Import library FastAPI untuk CORS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Create FastAPI app instance
app = FastAPI()

# Get origins from environment variable
origins = os.getenv("CORS_ORIGIN", "").split(",")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi dasar
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Koneksi ke MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Inisialisasi FastAPI
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Siapkan alat preprocessing Sastrawi
stemmer_factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_remover = stopword_factory.create_stop_word_remover()

# Variabel global untuk model yang sudah dilatih
vectorizer = None
knn_classifier = None
# Daftar kategori utama kita, sesuai dengan yang ada di notebook
categories = ["Ekonomi", "Hiburan", "Olahraga", "Politik", "Teknologi"]

# === Model Pydantic ===
# Model ini menentukan struktur data berita di database

class News(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    judul: str
    isi: str
    kategori: str
    confidence_score: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NewsCreate(BaseModel):
    # Model untuk data yang masuk dari frontend
    judul: str
    isi: str

class CategoryStats(BaseModel):
    # Model untuk data statistik di dashboard
    kategori: str
    jumlah: int

# === Fungsi Helper ===

def simplify_category(cat: str) -> str:
    """
    Fungsi helper untuk membersihkan data kategori dari CSV.
    Mengubah 'Nasional > Politik' menjadi 'Politik', dst.
    Sama seperti di notebook.
    """
    cat = str(cat) # Jaga-jaga kalau ada NaN
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
    """
    Fungsi untuk membersihkan teks: lowercase, hapus stopword, dan stemming.
    """
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# === Fungsi Model Machine Learning ===

async def train_model():
    """
    Fungsi ini mengambil SEMUA data dari database MongoDB,
    mempersiapkannya, dan melatih model TF-IDF + KNN.
    """
    global vectorizer, knn_classifier
    
    logging.info("Memulai pelatihan model...")
    
    # Ambil semua berita dari database
    all_news = await db.news.find({}, {"_id": 0}).to_list(10000)
    
    if len(all_news) < 5:
        # Kita butuh setidaknya beberapa data untuk melatih
        logging.warning("Data di database kurang dari 5, model mungkin tidak akurat.")
        if len(all_news) == 0:
            logging.error("Tidak ada data sama sekali. Model tidak bisa dilatih.")
            return

    # Siapkan data untuk training
    texts = []
    labels = []
    
    for news_item in all_news:
        # Gabungkan judul dan isi, lalu proses
        combined_text = news_item['judul'] + " " + news_item['isi']
        processed_text = preprocess_text(combined_text)
        texts.append(processed_text)
        labels.append(news_item['kategori'])
    
    # 1. Latih TF-IDF Vectorizer
    # Parameter ini sama dengan yang di notebook (max_features, ngram_range)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2)) # type: ignore
    X = vectorizer.fit_transform(texts)
    
    # 2. Latih KNN Classifier
    # Kita pakai min(5, len(texts)) agar aman jika data sangat sedikit
    n_neighbors = min(5, len(texts))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn_classifier.fit(X, labels)
    
    logging.info(f"Model berhasil dilatih dengan {len(texts)} data.")

def classify_news(judul: str, isi: str) -> tuple:
    """
    Menerima judul dan isi, lalu mengembalikan prediksi kategori
    dan seberapa yakin model tersebut (confidence score).
    """
    if vectorizer is None or knn_classifier is None:
        # Jika model belum siap (misalnya saat startup atau DB kosong)
        logging.warning("Model belum dilatih, klasifikasi dibatalkan.")
        return "Unknown", 0.0
    
    # Proses teks input sama seperti data training
    combined_text = judul + " " + isi
    processed_text = preprocess_text(combined_text)
    
    # Ubah teks jadi vektor TF-IDF
    X_new = vectorizer.transform([processed_text])
    
    # Prediksi probabilitas untuk tiap kelas
    probabilities = knn_classifier.predict_proba(X_new)[0]
    
    # Cari tahu kelas dengan probabilitas tertinggi
    max_prob_index = np.argmax(probabilities)
    predicted_category = knn_classifier.classes_[max_prob_index]
    confidence = float(np.max(probabilities))
    
    return predicted_category, confidence

# === Event Handler FastAPI ===

@app.on_event("startup")
async def startup_event():
    """
    Fungsi ini berjalan SATU KALI saat server pertama kali hidup.
    Tugasnya: mengecek database dan mengisinya jika kosong.
    """
    logging.info("Server startup...")
    count = await db.news.count_documents({})
    
    if count == 0:
        logging.warning("Database 'klasifikasi_berita' kosong. Memuat data awal dari CSV...")
        try:
            # 1. Baca CSV
            df = pd.read_csv("cnnindonesia_scraped.csv")
            
            # 2. Bersihkan data (sama seperti di notebook)
            df["judul"] = df["judul"].fillna("")
            df["isi"] = df["isi"].fillna("")
            df["kategori"] = df["kategori"].apply(simplify_category)
            
            # 3. Buang kategori 'Lainnya'
            df = df[df["kategori"].isin(categories)]
            
            # 4. Ambil 10 sampel acak per kategori (sesuai permintaan)
            initial_docs = []
            for cat in categories:
                # Ambil 10 sampel, atau kurang jika datanya tidak cukup
                sample_count = min(10, len(df[df['kategori'] == cat]))
                if sample_count > 0:
                    samples = df[df['kategori'] == cat].sample(n=sample_count, random_state=42)
                    
                    for _, row in samples.iterrows():
                        news_obj = News(
                            judul=row['judul'],
                            isi=row['isi'],
                            kategori=row['kategori'],
                            confidence_score=1.0 # Data awal kita anggap 100%
                        )
                        doc = news_obj.model_dump()
                        doc['created_at'] = doc['created_at'].isoformat()
                        initial_docs.append(doc)
            
            # 5. Masukkan ke database
            if initial_docs:
                await db.news.insert_many(initial_docs)
                logging.info(f"Berhasil memasukkan {len(initial_docs)} data awal ke database.")
            else:
                logging.error("Tidak ada data valid untuk dimasukkan dari CSV.")

        except FileNotFoundError:
            logging.error("ERROR: File 'cnnindonesia_scraped.csv' tidak ditemukan.")
            logging.error("Server akan berjalan tanpa data awal. Silakan tambahkan berita via API.")
        except Exception as e:
            logging.error(f"Gagal memuat data CSV: {e}")
    
    else:
        logging.info(f"Database sudah berisi {count} data. Startup normal.")
    
    # Setelah data siap (baik data lama atau baru), latih modelnya
    await train_model()

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
    """
    Endpoint untuk menerima berita baru dari frontend,
    mengklasifikasikannya, menyimpannya, dan melatih ulang model.
    """
    # 1. Klasifikasikan berita baru
    kategori, confidence = classify_news(input.judul, input.isi)
    
    # 2. Buat objek berita
    news_obj = News(
        judul=input.judul,
        isi=input.isi,
        kategori=kategori,
        confidence_score=confidence
    )
    
    # 3. Simpan ke database
    doc = news_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.news.insert_one(doc)
    
    # 4. Latih ulang model dengan data baru (secara asynchronous)
    # Tidak perlu 'await' agar frontend tidak menunggu lama
    # Tapi untuk proyek ini, 'await' memastikan data selalu update
    await train_model() 
    
    return news_obj

@api_router.get("/news", response_model=List[News])
async def get_all_news():
    """
    Endpoint untuk mengambil semua berita dari database,
    diurutkan dari yang terbaru.
    """
    news_list = await db.news.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    
    # Konversi string ISO kembali ke datetime untuk Pydantic
    for news_item in news_list:
        if isinstance(news_item['created_at'], str):
            news_item['created_at'] = datetime.fromisoformat(news_item['created_at'])
    
    return news_list

@api_router.get("/news/{kategori}", response_model=List[News])
async def get_news_by_category(kategori: str):
    """
    Endpoint untuk mengambil berita berdasarkan kategori.
    """
    if kategori not in categories:
        raise HTTPException(status_code=404, detail="Kategori tidak ditemukan")
        
    news_list = await db.news.find({"kategori": kategori}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    
    for news_item in news_list:
        if isinstance(news_item['created_at'], str):
            news_item['created_at'] = datetime.fromisoformat(news_item['created_at'])
    
    return news_list

@api_router.get("/categories/stats", response_model=List[CategoryStats])
async def get_category_stats():
    """
    Endpoint untuk statistik di dashboard frontend.
    """
    stats = []
    
    # Hitung jumlah dokumen untuk setiap kategori
    for kategori in categories:
        count = await db.news.count_documents({"kategori": kategori})
        stats.append(CategoryStats(kategori=kategori, jumlah=count))
    
    return stats

@api_router.post("/train")
async def retrain_model_manual():
    """
    Endpoint tambahan untuk memicu pelatihan ulang secara manual.
    """
    await train_model()
    return {"message": "Model retrained successfully"}

# === Konfigurasi Server ===

# Masukkan router API ke aplikasi utama
app.include_router(api_router)

# Konfigurasi CORS (Cross-Origin Resource Sharing)
# Ini penting agar React (localhost:3000) bisa 'bicara' dengan FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konfigurasi logging dasar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Jalankan server hanya jika file ini dieksekusi langsung
if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)