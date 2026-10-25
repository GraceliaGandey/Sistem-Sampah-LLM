"""
===========================================================================
 LANGKAH 1 — PENGUMPULAN DATA (Data Collection)
===========================================================================
 Tujuan : mengumpulkan minimal 500 teks opini masyarakat tentang layanan
          pengelolaan sampah (bank sampah, jemput sampah, daur ulang) secara OTOMATIS.

 Sumber data:
   (a) Ulasan Google Play dari aplikasi pengelolaan sampah -> scraping
   (b) Link website lain (berita, forum)               -> ekstraksi teks
   (c) Input manual (file CSV tambahan)                -> opsional

 Output : data/dataset_mentah.csv
 Kolom  : id, sumber, aplikasi, kota, teks, rating, tanggal, jumlah_like
===========================================================================
"""
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from google_play_scraper import Sort, reviews, search

import config


# ---------------------------------------------------------------------------
# (a) SCRAPING ULASAN GOOGLE PLAY
# ---------------------------------------------------------------------------
def ambil_ulasan_play_store(app_id, nama_aplikasi, kota, jumlah=300):
    """Mengambil ulasan terbaru dari satu aplikasi di Google Play.

    Library google-play-scraper bekerja seperti browser: membuka halaman
    ulasan aplikasi lalu membaca isinya. Ulasan diambil per 100 (satu
    "halaman"), lalu memakai continuation_token untuk halaman berikutnya.
    """
    semua_ulasan = []
    token_lanjutan = None

    while len(semua_ulasan) < jumlah:
        try:
            hasil, token_lanjutan = reviews(
                app_id,
                lang="id",            # ulasan berbahasa Indonesia
                country="id",         # dari Play Store Indonesia
                sort=Sort.NEWEST,     # urutkan dari yang terbaru
                count=min(100, jumlah - len(semua_ulasan)),
                continuation_token=token_lanjutan,
            )
        except Exception as error:
            print(f"   ! Gagal mengambil {app_id}: {error}")
            break

        semua_ulasan.extend(hasil)
        if not hasil or token_lanjutan is None:   # sudah tidak ada halaman lagi
            break
        time.sleep(1)  # sopan santun: jangan membanjiri server

    # Ambil kolom yang kita perlukan saja
    data = [{
        "sumber": "google_play",
        "aplikasi": nama_aplikasi,
        "kota": kota,
        "teks": u["content"],
        "rating": u["score"],          # bintang 1-5 (dipakai sebagai pembanding)
        "tanggal": u["at"],
        "jumlah_like": u["thumbsUpCount"],
    } for u in semua_ulasan]
    return data


def tebak_kota_dari_judul(judul):
    """Mencari nama kota di judul aplikasi, misal 'Bank Sampah Kota Malang' -> Malang."""
    for kota in config.KOORDINAT_KOTA:
        if kota.lower() in judul.lower():
            return kota
    return "Tidak diketahui"


def cari_aplikasi_air_otomatis():
    """Mencari aplikasi pengelolaan sampah lain di Play Store memakai kata kunci.

    Hasil pencarian disaring: hanya aplikasi yang judulnya mengandung
    kata seperti 'sampah', 'daur ulang', 'recycle', dll.
    """
    ditemukan = []
    for kata_kunci in config.KATA_KUNCI_PENCARIAN:
        try:
            hasil = search(kata_kunci, lang="id", country="id", n_hits=30)
        except Exception as error:
            print(f"   ! Pencarian '{kata_kunci}' gagal: {error}")
            continue
        for app in hasil:
            judul = app.get("title") or ""
            if app.get("appId") and any(p in judul.lower() for p in config.PENANDA_APLIKASI_SAMPAH):
                ditemukan.append({
                    "app_id": app["appId"],
                    "nama": judul,
                    "kota": tebak_kota_dari_judul(judul),
                })
    return ditemukan


# ---------------------------------------------------------------------------
# (b) EKSTRAKSI TEKS DARI LINK WEBSITE / LINK PLAY STORE
# ---------------------------------------------------------------------------
def ambil_teks_dari_website(url):
    """Mengambil paragraf teks dari halaman web (misal berita tentang sampah).

    BeautifulSoup membaca HTML, lalu kita ambil isi semua tag <p>.
    Paragraf yang terlalu pendek (menu, iklan) dibuang.
    """
    halaman = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(halaman.text, "html.parser")
    paragraf = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraf = [p for p in paragraf if len(p.split()) >= 8]
    return [{
        "sumber": "website", "aplikasi": url, "kota": "Tidak diketahui",
        "teks": p, "rating": None, "tanggal": None, "jumlah_like": None,
    } for p in paragraf]


def ambil_dari_link(url, kota="Tidak diketahui", jumlah=300):
    """Pintu masuk serbaguna: user cukup memasukkan LINK.

    - Link Google Play  -> scraping ulasan aplikasinya
    - Link lain         -> ekstraksi paragraf dari website
    """
    cocok = re.search(r"[?&]id=([\w\.]+)", url)
    if "play.google.com" in url and cocok:
        app_id = cocok.group(1)
        return ambil_ulasan_play_store(app_id, app_id, kota, jumlah)
    return ambil_teks_dari_website(url)


# ---------------------------------------------------------------------------
# FUNGSI UTAMA LANGKAH 1
# ---------------------------------------------------------------------------
def kumpulkan_dataset(link_tambahan=None, file_manual=None):
    """Menjalankan seluruh pengumpulan data lalu menyimpan CSV."""
    print("\n[LANGKAH 1] Pengumpulan data")

    # 1. Gabungkan aplikasi awal + hasil pencarian otomatis (tanpa duplikat)
    daftar_aplikasi = {a["app_id"]: a for a in config.APLIKASI_AWAL}
    for app in cari_aplikasi_air_otomatis():
        daftar_aplikasi.setdefault(app["app_id"], app)
    print(f"   Ditemukan {len(daftar_aplikasi)} aplikasi pengelolaan sampah")

    # 2. Scraping tiap aplikasi
    semua_data = []
    for app in daftar_aplikasi.values():
        data = ambil_ulasan_play_store(app["app_id"], app["nama"], app["kota"],
                                       config.JUMLAH_ULASAN_PER_APLIKASI)
        print(f"   - {app['nama'][:40]:40s} ({app['kota']}): {len(data)} ulasan")
        semua_data.extend(data)

    # 3. Link tambahan dari user (opsional)
    for url in (link_tambahan or []):
        data = ambil_dari_link(url)
        print(f"   - Link {url[:50]}: {len(data)} teks")
        semua_data.extend(data)

    df = pd.DataFrame(semua_data)

    # 4. Input manual (opsional): CSV dengan minimal kolom 'teks'
    if file_manual:
        manual = pd.read_csv(file_manual)
        manual["sumber"] = manual.get("sumber", "manual")
        df = pd.concat([df, manual], ignore_index=True)

    # 5. Beri nomor id unik untuk setiap teks
    df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(config.FILE_DATASET_MENTAH, index=False)

    print(f"   Total data mentah: {len(df)} -> {config.FILE_DATASET_MENTAH}")
    if len(df) < config.TARGET_MINIMAL_DATA:
        print(f"   ! PERHATIAN: data kurang dari {config.TARGET_MINIMAL_DATA}. "
              "Tambahkan aplikasi di config.APLIKASI_AWAL atau naikkan "
              "JUMLAH_ULASAN_PER_APLIKASI.")
    return df


if __name__ == "__main__":
    kumpulkan_dataset()
