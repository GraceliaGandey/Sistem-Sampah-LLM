# AI-Based Waste Management Sentiment and Community Intervention Recommendation System

LLM-Based Social Problem Analysis and Recommendation System (Exercise #1, Decision Making in AI).
Topik: Sampah Plastik & Pengelolaan Sampah. LLM: Google Gemini.
Data: ulasan Google Play aplikasi bank sampah / jemput sampah / daur ulang (Duitin, Rekosistem, Mallsampah, + pencarian otomatis).

## Cara menjalankan

```
pip install -r requirements.txt
$env:GEMINI_API_KEY="isi_key"            (PowerShell)   |  set GEMINI_API_KEY=isi_key  (CMD)
python main.py                             (langkah 1-6, pertama kali)
python main.py --mulai 3                   (melanjutkan analisis LLM tanpa scraping ulang)
python -m streamlit run dashboard.py       (dashboard + demo)
```

Laporan hasil: buka `output/laporan_hasil.html` di browser, lalu Ctrl+P, pilih "Save as PDF".
PENTING: jangan jalankan `python main.py` tanpa `--mulai` setelah langkah 3 berjalan, kecuali
`data/_checkpoint_llm.csv` dihapus dulu (scraping ulang mengubah nomor id data).

## Struktur

| File | Isi |
|---|---|
| config.py | Semua pengaturan (API key, model, daftar aplikasi sampah, kategori, batas data) |
| langkah1_pengumpulan_data.py | Scraping Google Play + ekstraksi teks dari link |
| langkah2_preprocessing.py | Pembersihan teks (2 versi: untuk LLM & untuk kata kunci) |
| langkah3_analisis_llm.py | Prompt + Gemini: sentimen, kategori masalah, confidence, alasan (XAI) |
| langkah4_eksperimen.py | Evaluasi keputusan AI: vs bintang, kasus sulit, prompt, konsistensi, confidence |
| langkah5_rca_rekomendasi.py | Prioritas, tren/anomali, RAG, RCA 5 Whys, fishbone, rekomendasi |
| langkah6_laporan.py | Grafik, peta, laporan HTML |
| main.py | Menjalankan semua langkah |
| dashboard.py | Dashboard Streamlit |
| data/kasus_uji_sulit.csv | 25 kasus uji: sarkasme, ambigu, campuran, negasi, dialek, multibahasa |
| data/pengetahuan_sampah.txt | Fakta resmi (SIPSN KLHK, Perpres 97/2017, KLH) untuk RAG |
