"""
===========================================================================
 LANGKAH 2 — PREPROCESSING (Pembersihan Teks)
===========================================================================
 Tujuan : membersihkan data mentah supaya keputusan AI lebih akurat.

 PENTING (bahan analisis "Kualitas dataset & preprocessing"):
 Kita membuat DUA versi teks, karena kebutuhan LLM dan kebutuhan
 analisis kata kunci itu BERBEDA.

  1) teks_llm    -> pembersihan RINGAN untuk dikirim ke LLM.
                    Kata "tidak", "tapi", "kok", tanda "?" "!" dan emoji
                    TETAP dipertahankan, karena kata-kata ini MENENTUKAN
                    sentimen. Contoh: "airnya TIDAK keruh" vs "airnya keruh".
                    Kalau "tidak" dibuang (stopword), maknanya terbalik!

  2) teks_bersih -> pembersihan PENUH ala materi kuliah (case folding,
                    hapus angka/tanda baca/emoji, normalisasi kata gaul,
                    stopword removal, stemming Sastrawi).
                    Dipakai untuk: frekuensi kata kunci, word cloud, RAG.

 Output : data/dataset_bersih.csv
===========================================================================
"""
import re

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import config

# ---------------------------------------------------------------------------
# Kamus normalisasi kata gaul / singkatan yang sering muncul di ulasan aplikasi sampah
# ---------------------------------------------------------------------------
KAMUS_NORMALISASI = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "tdk": "tidak", "g": "tidak", "enggak": "tidak", "kagak": "tidak",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "blm": "belum", "blum": "belum",
    "bgt": "banget", "bngt": "banget", "sy": "saya", "aq": "aku", "sya": "saya",
    "tp": "tapi", "tpi": "tapi", "jd": "jadi", "jdi": "jadi", "dr": "dari",
    "trs": "terus", "trus": "terus", "lg": "lagi", "aja": "saja", "aj": "saja",
    "bs": "bisa", "bsa": "bisa", "knp": "kenapa", "gmn": "bagaimana", "gimana": "bagaimana",
    "sm": "sama", "msh": "masih", "mati2": "mati", "ngadu": "mengadu",
    "apk": "aplikasi", "kg": "kilo", "kilo": "kilo", "jmpt": "jemput", "dtg": "datang", "dateng": "datang", "app": "aplikasi", "apl": "aplikasi", "plis": "tolong",
    "mantul": "mantap", "mksh": "terima kasih", "makasih": "terima kasih", "thx": "terima kasih",
}

# Kata yang JANGAN dibuang walaupun ada di daftar stopword Sastrawi,
# karena membawa makna sentimen / negasi.
KATA_PENTING = {"tidak", "belum", "bukan", "tapi", "tetapi", "namun", "kurang",
                "sangat", "banget", "jangan", "masih", "lagi"}

# Stopword tambahan khusus domain (sering muncul tapi tidak informatif)
STOPWORD_TAMBAHAN = {"duitin", "rekosistem", "mallsampah", "nya", "sih", "deh", "dong", "kak", "min", "admin", "ya", "yah"}

# Pola emoji (rentang kode Unicode emoji)
POLA_EMOJI = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF\U0001FA70-\U0001FAFF]+", flags=re.UNICODE)

# Siapkan stemmer & stopword Sastrawi sekali saja (lambat jika dibuat berulang)
STEMMER = StemmerFactory().create_stemmer()
STOPWORDS = (set(StopWordRemoverFactory().get_stop_words()) | STOPWORD_TAMBAHAN) - KATA_PENTING


# ---------------------------------------------------------------------------
# Fungsi-fungsi kecil (satu fungsi = satu tahap preprocessing)
# ---------------------------------------------------------------------------
def hapus_karakter_berulang(teks):
    """'mantaaaap' -> 'mantap', 'lamaaa' -> 'lama' (huruf diulang >2 kali)."""
    return re.sub(r"(\w)\1{2,}", r"\1", teks)


def normalisasi_kata(teks):
    """Mengganti kata gaul/singkatan dengan kata baku memakai kamus."""
    return " ".join(KAMUS_NORMALISASI.get(kata, kata) for kata in teks.split())


def bersihkan_untuk_llm(teks):
    """Pembersihan RINGAN: konteks & emoji dipertahankan untuk LLM."""
    teks = str(teks)
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)   # hapus URL
    teks = re.sub(r"\s+", " ", teks)                # rapikan spasi/enter
    teks = hapus_karakter_berulang(teks)
    return teks.strip()


def bersihkan_penuh(teks):
    """Pembersihan PENUH (sesuai materi kuliah) untuk analisis kata kunci."""
    teks = str(teks).lower()                         # 1. case folding
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)    # 2. hapus URL
    teks = POLA_EMOJI.sub(" ", teks)                 # 3. hapus emoji
    teks = re.sub(r"\d+", " ", teks)                 # 4. hapus angka
    teks = re.sub(r"[^\w\s]", " ", teks)             # 5. hapus tanda baca
    teks = hapus_karakter_berulang(teks)             # 6. huruf berulang
    teks = normalisasi_kata(teks)                    # 7. normalisasi kata gaul
    kata = [k for k in teks.split() if k not in STOPWORDS and len(k) > 1]   # 8. stopword
    return " ".join(STEMMER.stem(k) for k in kata)   # 9. stemming


# ---------------------------------------------------------------------------
# FUNGSI UTAMA LANGKAH 2
# ---------------------------------------------------------------------------
def preprocessing(df):
    print("\n[LANGKAH 2] Preprocessing")
    laporan = {"data_awal": len(df)}

    # a) Buang data kosong
    df = df.dropna(subset=["teks"])
    df = df[df["teks"].astype(str).str.strip() != ""]
    laporan["setelah_hapus_kosong"] = len(df)

    # b) Buang duplikat (ulasan yang sama persis, misal spam copy-paste)
    df = df.drop_duplicates(subset=["teks"])
    laporan["setelah_hapus_duplikat"] = len(df)

    # c) Buat dua versi teks
    df = df.copy()
    df["teks_llm"] = df["teks"].apply(bersihkan_untuk_llm)
    print("   Stemming dengan Sastrawi (bisa 1-3 menit)...")
    df["teks_bersih"] = df["teks"].apply(bersihkan_penuh)

    # d) Hitung panjang teks (ulasan 1-2 kata biasanya sulit dinilai)
    df["jumlah_kata"] = df["teks_llm"].str.split().str.len()
    df["teks_pendek"] = df["jumlah_kata"] <= 2
    laporan["teks_pendek_<=2_kata"] = int(df["teks_pendek"].sum())

    # e) Label pembanding dari rating bintang (BUKAN kebenaran mutlak!)
    #    1-2 = negatif, 3 = netral, 4-5 = positif
    def label_dari_bintang(r):
        if pd.isna(r):
            return None
        return "negatif" if r <= 2 else ("netral" if r == 3 else "positif")
    df["label_bintang"] = df["rating"].apply(label_dari_bintang)

    df.to_csv(config.FILE_DATASET_BERSIH, index=False)
    for k, v in laporan.items():
        print(f"   {k:28s}: {v}")
    print(f"   -> {config.FILE_DATASET_BERSIH}")
    return df, laporan


if __name__ == "__main__":
    preprocessing(pd.read_csv(config.FILE_DATASET_MENTAH))
