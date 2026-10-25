"""
===========================================================================
 main.py — MENJALANKAN SELURUH SISTEM DARI AWAL SAMPAI AKHIR
===========================================================================
 LLM-Based Social Problem Analysis and Recommendation System
 Topik: Sampah Plastik & Pengelolaan Sampah (AI-Based Waste Management
        Sentiment and Community Intervention Recommendation System)

 Alur sistem:
   Langkah 1  Pengumpulan data   (scraping Google Play / link website)
   Langkah 2  Preprocessing      (pembersihan teks)
   Langkah 3  Analisis LLM       (sentimen + kategori masalah + alasan)
   Langkah 4  Eksperimen         (evaluasi keputusan AI & kegagalan)
   Langkah 5  RCA & rekomendasi  (RAG + fishbone + rekomendasi)
   Langkah 6  Laporan            (grafik, peta, laporan HTML)

 Contoh pemakaian:
   python main.py                          # jalankan semua langkah
   python main.py --lewati-scraping        # pakai data/dataset_mentah.csv yang sudah ada
   python main.py --link "https://play.google.com/store/apps/details?id=com.ptam"
   python main.py --mulai 4                # mulai dari langkah 4 (pakai hasil_llm.csv)
 Lalu jalankan dashboard:  streamlit run dashboard.py
===========================================================================
"""
import argparse
import json
import os

import pandas as pd

import config
from langkah1_pengumpulan_data import kumpulkan_dataset
from langkah2_preprocessing import preprocessing
from langkah3_analisis_llm import analisis_dataset
from langkah4_eksperimen import jalankan_semua_eksperimen
from langkah5_rca_rekomendasi import jalankan_rca
from langkah6_laporan import jalankan_laporan

FILE_LAPORAN_PRE = os.path.join(config.FOLDER_OUTPUT, "laporan_preprocessing.json")


def main():
    parser = argparse.ArgumentParser(description="Sistem analisis pengelolaan sampah berbasis LLM")
    parser.add_argument("--lewati-scraping", action="store_true", help="pakai dataset_mentah.csv yang sudah ada")
    parser.add_argument("--link", nargs="*", default=[], help="link Google Play / website tambahan")
    parser.add_argument("--file-manual", default=None, help="CSV input manual (kolom minimal: teks)")
    parser.add_argument("--mulai", type=int, default=1, help="mulai dari langkah ke- (1-6)")
    args = parser.parse_args()

    # ---- Langkah 1 & 2 ----
    if args.mulai <= 2:
        if args.mulai == 1 and not args.lewati_scraping:
            df_mentah = kumpulkan_dataset(args.link, args.file_manual)
        else:
            df_mentah = pd.read_csv(config.FILE_DATASET_MENTAH)
        df_bersih, laporan_pre = preprocessing(df_mentah)
        json.dump(laporan_pre, open(FILE_LAPORAN_PRE, "w"), indent=2)
    else:
        laporan_pre = json.load(open(FILE_LAPORAN_PRE)) if os.path.exists(FILE_LAPORAN_PRE) else {}

    # ---- Langkah 3 ----
    if args.mulai <= 3:
        df_bersih = pd.read_csv(config.FILE_DATASET_BERSIH)
        df = analisis_dataset(df_bersih)
    else:
        df = pd.read_csv(config.FILE_HASIL_LLM)

    # ---- Langkah 4, 5, 6 ----
    if args.mulai <= 4:
        jalankan_semua_eksperimen(df)
    if args.mulai <= 5:
        jalankan_rca(df)
    jalankan_laporan(df, laporan_pre)

    print("\nSELESAI. File untuk dikumpulkan:")
    print(f"  - Dataset CSV : {config.FILE_HASIL_LLM}")
    print(f"  - Laporan     : {config.FOLDER_OUTPUT}/laporan_hasil.html  (cetak ke PDF)")
    print("  - Dashboard   : streamlit run dashboard.py")


if __name__ == "__main__":
    main()
