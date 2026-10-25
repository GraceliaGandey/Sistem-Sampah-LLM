"""
===========================================================================
 config.py  —  PENGATURAN SISTEM (semua "tombol" ada di sini)
===========================================================================
 Topik  : AI-Based Waste Management Sentiment and Community
          Intervention Recommendation System (Sampah Plastik & Pengelolaan Sampah)

 File ini TIDAK berisi logika program, hanya pengaturan.
 Kalau mau mengganti model LLM, jumlah data, atau daftar aplikasi pengelolaan sampah,
 cukup ubah di sini, file lain otomatis ikut.
===========================================================================
"""
import os

# ---------------------------------------------------------------------------
# 0. IDENTITAS KELOMPOK (muncul di laporan otomatis)
# ---------------------------------------------------------------------------
ANGGOTA_KELOMPOK = [
    "Nama Anggota 1 (NIM)",
    "Anggota 2 (NIM)",
    "Anggota 3 (NIM)",
]

# ---------------------------------------------------------------------------
# 1. API KEY LLM
# ---------------------------------------------------------------------------
# Ambil API key gratis di: https://aistudio.google.com/apikey
# Cara paling aman: simpan di environment variable, jangan ditulis di kode
# yang dikumpulkan ke dosen.
#   Windows (CMD)   : set GEMINI_API_KEY=isi_key_kalian
#   Linux/Mac/Colab : export GEMINI_API_KEY=isi_key_kalian
#   Colab (alternatif): os.environ["GEMINI_API_KEY"] = "isi_key_kalian"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Model Gemini yang dipakai. Cek daftar model terbaru di
# https://ai.google.dev/gemini-api/docs/models  (kalau model ini sudah
# tidak tersedia, ganti saja namanya, misal "gemini-3.6-flash").
MODEL_GEMINI = "gemini-3.8-flash"

# ---------------------------------------------------------------------------
# 2. MODEL PEMBANDING (OPSIONAL, untuk BONUS "Gemini vs LLM lain")
# ---------------------------------------------------------------------------
# Memakai layanan yang "OpenAI-compatible" (Groq, DeepSeek, OpenAI, dll).
# Groq punya tier gratis: https://console.groq.com/keys
# Kalau PEMBANDING_API_KEY kosong, perbandingan model otomatis dilewati.
# Butuh library tambahan: pip install openai  (tidak wajib)
PEMBANDING_API_KEY = os.environ.get("PEMBANDING_API_KEY", "")
PEMBANDING_BASE_URL = "https://api.groq.com/openai/v1"   # DeepSeek: https://api.deepseek.com
PEMBANDING_MODEL = "llama-3.3-70b-versatile"             # DeepSeek: "deepseek-chat"
PEMBANDING_NAMA = "Llama (Groq)"

# ---------------------------------------------------------------------------
# 3. PENGUMPULAN DATA
# ---------------------------------------------------------------------------
TARGET_MINIMAL_DATA = 500          # syarat tugas: minimal 500 teks
JUMLAH_ULASAN_PER_APLIKASI = 800   # maksimal ulasan yang diambil per aplikasi

# Aplikasi pengelolaan sampah / bank sampah digital yang sudah dicek ada di
# Google Play (app_id = bagian "id=..." pada link Play Store).
# "kota" = wilayah layanan utama aplikasi (dipakai untuk analisis wilayah & peta).
APLIKASI_AWAL = [
    {"app_id": "com.duitin.duitincontributorapp", "nama": "Duitin (daur ulang)",          "kota": "Multi-kota"},
    {"app_id": "com.rekosistem.mobile",           "nama": "Rekosistem",                   "kota": "Jakarta"},
    {"app_id": "mallsampah.com.dev.mallsampah",   "nama": "Mallsampah - Recycling Network", "kota": "Makassar"},
]

# Sistem juga MENCARI aplikasi sampah lain secara otomatis di Play Store
# memakai kata kunci berikut (supaya data mencapai 500+).
KATA_KUNCI_PENCARIAN = ["bank sampah", "daur ulang sampah", "jemput sampah", "aplikasi sampah"]

# Kata yang harus ada di judul aplikasi supaya dianggap aplikasi pengelolaan sampah.
PENANDA_APLIKASI_SAMPAH = ["sampah", "daur ulang", "recycl", "waste", "rongsok", "octopus"]

# Koordinat (lintang, bujur) kota untuk PETA (bonus "Pemetaan menggunakan map").
# Kota baru dari pencarian otomatis dikenali kalau namanya ada di judul aplikasi.
KOORDINAT_KOTA = {
    "Jakarta": (-6.2088, 106.8456), "Bogor": (-6.5950, 106.8167), "Depok": (-6.4025, 106.7942),
    "Tangerang": (-6.1783, 106.6319), "Bekasi": (-6.2383, 106.9756), "Bandung": (-6.9175, 107.6191),
    "Cirebon": (-6.7320, 108.5523), "Semarang": (-6.9667, 110.4167), "Solo": (-7.5755, 110.8243),
    "Surakarta": (-7.5755, 110.8243), "Yogyakarta": (-7.7956, 110.3695), "Pati": (-6.7559, 111.0380),
    "Surabaya": (-7.2575, 112.7521), "Sidoarjo": (-7.4478, 112.7183), "Gresik": (-7.1539, 112.6561),
    "Malang": (-7.9666, 112.6326), "Denpasar": (-8.6705, 115.2126), "Badung": (-8.5819, 115.1771),
    "Mataram": (-8.5833, 116.1167), "Kupang": (-10.1772, 123.6070), "Medan": (3.5952, 98.6722),
    "Padang": (-0.9471, 100.4172), "Pekanbaru": (0.5071, 101.4478), "Palembang": (-2.9761, 104.7754),
    "Jambi": (-1.6101, 103.6131), "Lampung": (-5.3971, 105.2668), "Batam": (1.0456, 104.0305),
    "Pontianak": (-0.0263, 109.3425), "Banjarmasin": (-3.3186, 114.5944), "Banjarbaru": (-3.4425, 114.8301),
    "Sampit": (-2.5370, 112.9500), "Palangka Raya": (-2.2096, 113.9108), "Balikpapan": (-1.2379, 116.8529),
    "Samarinda": (-0.5022, 117.1536), "Tarakan": (3.3000, 117.6333), "Makassar": (-5.1477, 119.4327),
    "Manado": (1.4748, 124.8421), "Gorontalo": (0.5435, 123.0568), "Palu": (-0.8917, 119.8707),
    "Kendari": (-3.9985, 122.5129), "Ambon": (-3.6954, 128.1814), "Jayapura": (-2.5337, 140.7181),
    "Nasional": (-2.5, 118.0), "Multi-kota": (-2.5, 118.0),
}

# ---------------------------------------------------------------------------
# 4. ANALISIS LLM
# ---------------------------------------------------------------------------
UKURAN_BATCH = 100     # berapa ulasan dikirim ke LLM dalam 1 kali panggilan (besar = hemat kuota harian)
JEDA_DETIK = 4         # jeda antar panggilan supaya tidak kena batas kuota gratis
MAKS_ULANG = 4         # berapa kali mencoba ulang kalau API error

# Batas jumlah teks yang dianalisis LLM (syarat tugas: 500-1000).
# Kalau data bersih lebih banyak, diambil sampel acak sebanyak ini.
# Isi None untuk menganalisis SEMUA data (lebih lama & lebih boros kuota).
MAKS_DATA_ANALISIS = 800

# Tingkat "berpikir" Gemini sebelum menjawab: "minimal", "low", "medium", "high".
# Klasifikasi sentimen tidak butuh berpikir panjang -> "low" jauh lebih cepat.
# Isi None untuk memakai bawaan model.
TINGKAT_BERPIKIR = "low"

# Label sentimen yang boleh dipakai LLM
LABEL_SENTIMEN = ["positif", "negatif", "netral"]

# Kategori masalah pengelolaan sampah (untuk "Problem Analysis").
# Mengikuti arahan topik: kebersihan, fasilitas, perilaku membuang sampah,
# dan hambatan partisipasi masyarakat.
KATEGORI_MASALAH = [
    "pengangkutan_sampah",   # jemput telat/tidak datang, jadwal tidak jelas, kurir
    "kebersihan_lingkungan", # sampah menumpuk, bau, TPS penuh, sampah liar
    "fasilitas_sampah",      # drop point/bank sampah jauh, tempat sampah kurang
    "perilaku_pemilahan",    # pemilahan, edukasi, kebiasaan membuang sampah
    "hambatan_partisipasi",  # minimal berat, area belum terjangkau, syarat rumit
    "harga_insentif",        # harga per kg rendah, poin/saldo, pencairan dana
    "pelayanan_cs",          # customer service lambat, komplain diabaikan
    "aplikasi_teknis",       # aplikasi error, login gagal, fitur tidak jalan
    "pujian_layanan",        # ulasan positif / apresiasi
    "lainnya",
]

# ---------------------------------------------------------------------------
# 5. EKSPERIMEN (Decision Making Analysis)
# ---------------------------------------------------------------------------
JUMLAH_SAMPEL_EKSPERIMEN_PROMPT = 60   # ulasan untuk membandingkan 2 versi prompt
JUMLAH_SAMPEL_KONSISTENSI = 30         # ulasan yang dianalisis berulang-ulang
JUMLAH_PENGULANGAN_KONSISTENSI = 3     # berapa kali diulang
SEED_ACAK = 42                         # supaya sampel acak selalu sama

# ---------------------------------------------------------------------------
# 6. LOKASI FILE
# ---------------------------------------------------------------------------
# Folder dihitung dari lokasi file ini, jadi program bisa dijalankan dari mana saja.
FOLDER_PROYEK = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = os.path.join(FOLDER_PROYEK, "data")
FOLDER_OUTPUT = os.path.join(FOLDER_PROYEK, "output")

FILE_DATASET_MENTAH = f"{FOLDER_DATA}/dataset_mentah.csv"
FILE_DATASET_BERSIH = f"{FOLDER_DATA}/dataset_bersih.csv"
FILE_HASIL_LLM = f"{FOLDER_DATA}/hasil_llm.csv"           # dataset final + label LLM
FILE_KASUS_SULIT = f"{FOLDER_DATA}/kasus_uji_sulit.csv"
FILE_PENGETAHUAN = f"{FOLDER_DATA}/pengetahuan_sampah.txt"   # untuk RAG

os.makedirs(FOLDER_DATA, exist_ok=True)
os.makedirs(FOLDER_OUTPUT, exist_ok=True)
