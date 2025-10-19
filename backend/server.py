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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pickle
import numpy as np

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Indonesian text preprocessing
stemmer_factory = StemmerFactory()
stopword_factory = StopWordRemoverFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_remover = stopword_factory.create_stop_word_remover()

# Global variables for ML model
vectorizer = None
knn_classifier = None
categories = ["Politik", "Olahraga", "Teknologi", "Hiburan", "Ekonomi"]

# Sample dataset Indonesia
TRAINING_DATA = [
    # Politik
    {"judul": "Presiden Lantik Menteri Baru di Istana", "isi": "Presiden melantik menteri kabinet baru dalam upacara resmi di Istana Negara Jakarta. Pelantikan dihadiri oleh pejabat tinggi negara dan tokoh politik.", "kategori": "Politik"},
    {"judul": "DPR Setujui RUU Omnibus Law", "isi": "Dewan Perwakilan Rakyat menyetujui rancangan undang-undang omnibus law setelah pembahasan panjang dengan pemerintah dan berbagai fraksi.", "kategori": "Politik"},
    {"judul": "Partai Koalisi Gelar Rapat Koordinasi", "isi": "Partai politik koalisi pemerintah menggelar rapat koordinasi membahas strategi politik menjelang pemilu mendatang di Jakarta.", "kategori": "Politik"},
    {"judul": "Gubernur Kunjungi Daerah Terdampak Banjir", "isi": "Gubernur melakukan kunjungan ke daerah terdampak banjir untuk meninjau kondisi dan memberikan bantuan kepada korban bencana.", "kategori": "Politik"},
    {"judul": "Menteri Luar Negeri Hadiri KTT ASEAN", "isi": "Menteri Luar Negeri Indonesia menghadiri Konferensi Tingkat Tinggi ASEAN untuk membahas kerja sama regional dan isu geopolitik.", "kategori": "Politik"},
    {"judul": "Komisi Pemberantasan Korupsi Tangkap Pejabat", "isi": "KPK melakukan operasi tangkap tangan terhadap pejabat pemerintah yang diduga menerima suap dalam proyek infrastruktur.", "kategori": "Politik"},
    {"judul": "Pemilu Kepala Daerah Digelar Serentak", "isi": "Pemilihan kepala daerah akan digelar serentak di berbagai wilayah Indonesia dengan melibatkan jutaan pemilih.", "kategori": "Politik"},
    {"judul": "Anggaran Negara Tahun Depan Disahkan", "isi": "Pemerintah dan DPR resmi mengesahkan anggaran pendapatan dan belanja negara untuk tahun fiskal mendatang.", "kategori": "Politik"},
    {"judul": "Mahkamah Konstitusi Putuskan Sengketa Pilkada", "isi": "Mahkamah Konstitusi memutuskan sengketa hasil pemilihan kepala daerah setelah mendengar keterangan para pihak.", "kategori": "Politik"},
    {"judul": "Pemerintah Terbitkan Kebijakan Subsidi Baru", "isi": "Pemerintah menerbitkan kebijakan subsidi baru untuk membantu masyarakat menghadapi kenaikan harga kebutuhan pokok.", "kategori": "Politik"},
    
    # Olahraga
    {"judul": "Timnas Indonesia Menang 3-1 atas Thailand", "isi": "Tim nasional sepak bola Indonesia meraih kemenangan telak 3-1 melawan Thailand dalam laga persahabatan di Stadion Gelora Bung Karno.", "kategori": "Olahraga"},
    {"judul": "Atlet Bulu Tangkis Juara Turnamen Internasional", "isi": "Atlet bulu tangkis Indonesia berhasil menjuarai turnamen internasional setelah mengalahkan petarung dari China di partai final.", "kategori": "Olahraga"},
    {"judul": "Pelatih Baru Persib Bandung Resmi Diumumkan", "isi": "Manajemen Persib Bandung resmi mengumumkan pelatih baru yang akan menangani tim untuk musim kompetisi mendatang.", "kategori": "Olahraga"},
    {"judul": "Indonesia Raih Medali Emas SEA Games", "isi": "Kontingen Indonesia meraih medali emas dalam cabang atletik nomor lari 100 meter putra pada ajang SEA Games.", "kategori": "Olahraga"},
    {"judul": "Pemain Muda Timnas Cetak Hattrick", "isi": "Pemain muda timnas Indonesia mencetak tiga gol atau hattrick dalam pertandingan kualifikasi Piala Asia melawan tim tamu.", "kategori": "Olahraga"},
    {"judul": "Liga Indonesia Musim Baru Segera Dimulai", "isi": "Kompetisi Liga 1 Indonesia akan segera dimulai dengan diikuti 18 klub peserta dari seluruh Indonesia.", "kategori": "Olahraga"},
    {"judul": "Petinju Indonesia Menang TKO Ronde Ketiga", "isi": "Petinju Indonesia meraih kemenangan technical knockout di ronde ketiga melawan lawannya dalam pertarungan tinju profesional.", "kategori": "Olahraga"},
    {"judul": "Atlet Renang Pecahkan Rekor Nasional", "isi": "Atlet renang Indonesia memecahkan rekor nasional dalam nomor 200 meter gaya bebas pada kejuaraan renang nasional.", "kategori": "Olahraga"},
    {"judul": "Tim Basket Putri Lolos ke Final", "isi": "Tim basket putri Indonesia berhasil lolos ke babak final setelah mengalahkan Malaysia dalam pertandingan semifinal.", "kategori": "Olahraga"},
    {"judul": "Pembalap MotoGP Indonesia Finish di Posisi 5", "isi": "Pembalap MotoGP Indonesia berhasil finish di posisi kelima pada seri Grand Prix yang berlangsung di sirkuit internasional.", "kategori": "Olahraga"},
    
    # Teknologi
    {"judul": "Startup Indonesia Raih Pendanaan 100 Miliar", "isi": "Perusahaan startup teknologi Indonesia berhasil mendapatkan pendanaan seri B senilai 100 miliar rupiah dari investor asing.", "kategori": "Teknologi"},
    {"judul": "Peluncuran Smartphone 5G Terbaru", "isi": "Perusahaan teknologi meluncurkan smartphone 5G terbaru dengan fitur kamera canggih dan prosesor berkecepatan tinggi di Jakarta.", "kategori": "Teknologi"},
    {"judul": "Aplikasi E-Commerce Tambah Fitur AI", "isi": "Platform e-commerce terkemuka menambahkan fitur kecerdasan buatan untuk meningkatkan pengalaman berbelanja pengguna.", "kategori": "Teknologi"},
    {"judul": "Pemerintah Luncurkan Program Literasi Digital", "isi": "Pemerintah meluncurkan program literasi digital untuk meningkatkan kemampuan masyarakat dalam menggunakan teknologi.", "kategori": "Teknologi"},
    {"judul": "Jaringan Internet 5G Meluas ke Kota Besar", "isi": "Operator telekomunikasi memperluas jangkauan jaringan internet 5G ke berbagai kota besar di Indonesia.", "kategori": "Teknologi"},
    {"judul": "Perusahaan Teknologi Buka Kantor Baru", "isi": "Perusahaan teknologi global membuka kantor regional baru di Indonesia untuk ekspansi bisnis di Asia Tenggara.", "kategori": "Teknologi"},
    {"judul": "Aplikasi Fintech Capai 10 Juta Pengguna", "isi": "Aplikasi financial technology Indonesia mencapai 10 juta pengguna aktif setelah dua tahun beroperasi.", "kategori": "Teknologi"},
    {"judul": "Sistem Pembayaran Digital Terintegrasi", "isi": "Bank Indonesia meluncurkan sistem pembayaran digital terintegrasi untuk memudahkan transaksi antar platform.", "kategori": "Teknologi"},
    {"judul": "Robot AI Bantu Layanan Kesehatan", "isi": "Rumah sakit di Jakarta menggunakan robot berbasis AI untuk membantu tenaga medis dalam layanan kesehatan.", "kategori": "Teknologi"},
    {"judul": "Platform Edtech Tawarkan Kursus Gratis", "isi": "Platform pendidikan teknologi menawarkan ribuan kursus online gratis untuk pelajar dan mahasiswa Indonesia.", "kategori": "Teknologi"},
    
    # Hiburan
    {"judul": "Film Indonesia Raih Penghargaan Internasional", "isi": "Film Indonesia berhasil meraih penghargaan Best Picture dalam festival film internasional di Eropa.", "kategori": "Hiburan"},
    {"judul": "Konser Musik Band Legendaris Sold Out", "isi": "Konser musik band legendaris Indonesia berhasil sold out dalam waktu singkat dengan ribuan penonton hadir.", "kategori": "Hiburan"},
    {"judul": "Artis Indonesia Rilis Album Baru", "isi": "Penyanyi terkenal Indonesia merilis album musik terbaru yang langsung mendapat sambutan positif dari penggemar.", "kategori": "Hiburan"},
    {"judul": "Serial Drama Televisi Pecahkan Rating", "isi": "Serial drama televisi Indonesia pecahkan rating tertinggi dengan jutaan penonton setiap episodenya.", "kategori": "Hiburan"},
    {"judul": "Festival Film Dokumenter Digelar di Jakarta", "isi": "Festival film dokumenter internasional digelar di Jakarta menampilkan karya sineas dari berbagai negara.", "kategori": "Hiburan"},
    {"judul": "Penyanyi Indonesia Kolaborasi dengan Artis Luar", "isi": "Penyanyi pop Indonesia berkolaborasi dengan artis internasional dalam single musik terbaru mereka.", "kategori": "Hiburan"},
    {"judul": "Teater Musikal Broadway Tampil di Indonesia", "isi": "Pertunjukan teater musikal terkenal dari Broadway tampil perdana di Indonesia dengan pementasan spektakuler.", "kategori": "Hiburan"},
    {"judul": "Komika Stand Up Comedy Gelar Tur Nasional", "isi": "Komika stand up comedy terkenal menggelar tur nasional menghibur penonton di berbagai kota Indonesia.", "kategori": "Hiburan"},
    {"judul": "Game Show Televisi Hadirkan Format Baru", "isi": "Stasiun televisi meluncurkan game show dengan format baru yang interaktif melibatkan penonton di rumah.", "kategori": "Hiburan"},
    {"judul": "Aktor Indonesia Main Film Hollywood", "isi": "Aktor Indonesia mendapat kesempatan bermain dalam film produksi Hollywood yang akan rilis tahun depan.", "kategori": "Hiburan"},
    
    # Ekonomi
    {"judul": "Bank Sentral Naikkan Suku Bunga Acuan", "isi": "Bank Indonesia menaikkan suku bunga acuan untuk mengendalikan inflasi dan menjaga stabilitas ekonomi nasional.", "kategori": "Ekonomi"},
    {"judul": "Bursa Efek Indonesia Menguat 2 Persen", "isi": "Indeks Harga Saham Gabungan Bursa Efek Indonesia menguat 2 persen didorong oleh sentimen positif investor.", "kategori": "Ekonomi"},
    {"judul": "Harga Minyak Dunia Naik Tajam", "isi": "Harga minyak mentah dunia naik tajam akibat ketegangan geopolitik yang mempengaruhi pasokan global.", "kategori": "Ekonomi"},
    {"judul": "Inflasi Bulanan Tercatat 0.5 Persen", "isi": "Badan Pusat Statistik mencatat inflasi bulanan sebesar 0.5 persen akibat kenaikan harga bahan pangan.", "kategori": "Ekonomi"},
    {"judul": "Pemerintah Targetkan Pertumbuhan Ekonomi 5 Persen", "isi": "Pemerintah menargetkan pertumbuhan ekonomi nasional sebesar 5 persen untuk tahun fiskal mendatang.", "kategori": "Ekonomi"},
    {"judul": "Ekspor Indonesia Meningkat 10 Persen", "isi": "Nilai ekspor Indonesia meningkat 10 persen didorong oleh permintaan komoditas dari negara-negara Asia.", "kategori": "Ekonomi"},
    {"judul": "Rupiah Menguat Terhadap Dolar AS", "isi": "Nilai tukar rupiah menguat terhadap dolar Amerika Serikat seiring membaiknya kondisi ekonomi domestik.", "kategori": "Ekonomi"},
    {"judul": "Perusahaan Teknologi IPO di Bursa Saham", "isi": "Perusahaan teknologi Indonesia melakukan penawaran umum perdana saham di Bursa Efek Indonesia.", "kategori": "Ekonomi"},
    {"judul": "Harga Emas Dunia Tembus Rekor Baru", "isi": "Harga emas dunia tembus rekor tertinggi sepanjang masa akibat ketidakpastian ekonomi global.", "kategori": "Ekonomi"},
    {"judul": "Investasi Asing Masuk Triliunan Rupiah", "isi": "Indonesia menerima investasi asing langsung senilai triliunan rupiah untuk proyek infrastruktur dan industri.", "kategori": "Ekonomi"},
]

# Define Models
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

def preprocess_text(text: str) -> str:
    """Preprocess Indonesian text: lowercase, remove stopwords, stemming"""
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

async def train_model():
    """Train KNN model with current dataset"""
    global vectorizer, knn_classifier
    
    # Get all news from database
    all_news = await db.news.find({}, {"_id": 0}).to_list(10000)
    
    # If no data in DB, use training data
    if len(all_news) < 5:
        all_news = TRAINING_DATA
    
    # Prepare training data
    texts = []
    labels = []
    
    for news_item in all_news:
        combined_text = news_item['judul'] + " " + news_item['isi']
        processed_text = preprocess_text(combined_text)
        texts.append(processed_text)
        labels.append(news_item['kategori'])
    
    # Train TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    
    # Train KNN classifier
    n_neighbors = min(5, len(texts))
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn_classifier.fit(X, labels)
    
    logging.info(f"Model trained with {len(texts)} samples")

def classify_news(judul: str, isi: str) -> tuple:
    """Classify news and return category with confidence score"""
    if vectorizer is None or knn_classifier is None:
        return "Unknown", 0.0
    
    # Preprocess input
    combined_text = judul + " " + isi
    processed_text = preprocess_text(combined_text)
    
    # Vectorize
    X = vectorizer.transform([processed_text])
    
    # Predict
    predicted_category = knn_classifier.predict(X)[0]
    
    # Get probability/confidence
    probabilities = knn_classifier.predict_proba(X)[0]
    confidence = float(max(probabilities))
    
    return predicted_category, confidence

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    # Insert training data if collection is empty
    count = await db.news.count_documents({})
    if count == 0:
        # Prepare training documents
        training_docs = []
        for item in TRAINING_DATA:
            news_obj = News(
                judul=item['judul'],
                isi=item['isi'],
                kategori=item['kategori'],
                confidence_score=1.0
            )
            doc = news_obj.model_dump()
            doc['created_at'] = doc['created_at'].isoformat()
            training_docs.append(doc)
        
        await db.news.insert_many(training_docs)
        logging.info(f"Inserted {len(training_docs)} training samples")
    
    # Train model
    await train_model()
    logging.info("Model initialized and trained")

@api_router.get("/")
async def root():
    return {"message": "News Classifier API"}

@api_router.post("/news", response_model=News)
async def create_news(input: NewsCreate):
    """Add new news and classify it automatically"""
    # Classify the news
    kategori, confidence = classify_news(input.judul, input.isi)
    
    # Create news object
    news_obj = News(
        judul=input.judul,
        isi=input.isi,
        kategori=kategori,
        confidence_score=confidence
    )
    
    # Save to database
    doc = news_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.news.insert_one(doc)
    
    # Retrain model with new data
    await train_model()
    
    return news_obj

@api_router.get("/news", response_model=List[News])
async def get_all_news():
    """Get all news"""
    news_list = await db.news.find({}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for news_item in news_list:
        if isinstance(news_item['created_at'], str):
            news_item['created_at'] = datetime.fromisoformat(news_item['created_at'])
    
    return news_list

@api_router.get("/news/{kategori}", response_model=List[News])
async def get_news_by_category(kategori: str):
    """Get news by category"""
    news_list = await db.news.find({"kategori": kategori}, {"_id": 0}).sort("created_at", -1).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for news_item in news_list:
        if isinstance(news_item['created_at'], str):
            news_item['created_at'] = datetime.fromisoformat(news_item['created_at'])
    
    return news_list

@api_router.get("/categories/stats", response_model=List[CategoryStats])
async def get_category_stats():
    """Get statistics per category"""
    stats = []
    
    for kategori in categories:
        count = await db.news.count_documents({"kategori": kategori})
        stats.append(CategoryStats(kategori=kategori, jumlah=count))
    
    return stats

@api_router.post("/train")
async def retrain_model():
    """Manually retrain the model"""
    await train_model()
    return {"message": "Model retrained successfully"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()