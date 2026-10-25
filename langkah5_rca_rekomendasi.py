"""
===========================================================================
 LANGKAH 5 — SOCIAL PROBLEM ANALYSIS, RCA, FISHBONE & REKOMENDASI
===========================================================================
 Alur:
  5.1 Ringkas data       : hitung masalah terbanyak, per kota, kata dominan
  5.2 Skor prioritas     : FREKUENSI x DAMPAK (aturan transparan, bisa dihitung manual)
  5.3 Tren & anomali     : minggu dengan lonjakan keluhan yang tidak wajar
  5.4 RAG                : cari fakta resmi (SIPSN KLHK, Perpres Jakstranas, KLH) yang
                           relevan dari file pengetahuan_air_bersih.txt
  5.5 LLM sebagai analis : data + fakta -> RCA (5 Whys), Fishbone, Rekomendasi
  5.6 Gambar fishbone    : output/fishbone.png

 Output : output/rca_rekomendasi.json, output/fishbone.png, CSV pendukung
===========================================================================
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")                       # supaya bisa menggambar tanpa layar (Colab/server)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from langkah3_analisis_llm import baca_json, tanya_llm

# Dampak tiap kategori terhadap masyarakat (1.0 = paling berat).
# Ini KEPUTUSAN KELOMPOK -> jelaskan alasannya saat presentasi.
BOBOT_DAMPAK = {
    "kebersihan_lingkungan": 1.0,  # sampah menumpuk = risiko kesehatan & lingkungan langsung
    "pengangkutan_sampah": 0.9,    # sampah tidak terangkut -> menumpuk / dibuang sembarangan
    "hambatan_partisipasi": 0.8,   # warga mau ikut tapi terhalang -> daur ulang gagal
    "fasilitas_sampah": 0.8,       # tanpa fasilitas, pemilahan tidak berlanjut
    "perilaku_pemilahan": 0.7,     # akar masalah jangka panjang
    "harga_insentif": 0.5,         # memengaruhi motivasi, bukan kebutuhan dasar
    "pelayanan_cs": 0.5,           # memperparah masalah lain
    "aplikasi_teknis": 0.4,        # mengganggu, tapi ada jalur lain
    "lainnya": 0.3,
}

KATEGORI_FISHBONE = ["Infrastruktur & Fasilitas", "Layanan Pengangkutan",
                     "Perilaku Masyarakat", "Kebijakan & Regulasi",
                     "Teknologi & Aplikasi", "Ekonomi & Insentif"]


def simpan_csv(df, nama):
    df.to_csv(os.path.join(config.FOLDER_OUTPUT, nama), index=False)


# ---------------------------------------------------------------------------
# 5.1 RINGKASAN DATA
# ---------------------------------------------------------------------------
def kata_dominan(daftar_teks, n=8):
    """Menghitung kata yang paling sering muncul (dari teks_bersih)."""
    hitung = Counter(" ".join(daftar_teks.fillna("")).split())
    return [k for k, _ in hitung.most_common(n)]


def ringkas_data(df):
    neg = df[df["sentimen"] == "negatif"]

    per_kategori = (neg.groupby("kategori_masalah")
                    .agg(jumlah_keluhan=("id", "count"), rata_confidence=("confidence", "mean"),
                         total_like=("jumlah_like", "sum"))
                    .reset_index())
    per_kategori["persen_dari_keluhan"] = (per_kategori["jumlah_keluhan"] / max(len(neg), 1)).round(3)
    per_kategori["kata_dominan"] = [", ".join(kata_dominan(neg[neg["kategori_masalah"] == k]["teks_bersih"]))
                                    for k in per_kategori["kategori_masalah"]]

    per_kota = (df.groupby("kota")
                .agg(jumlah_ulasan=("id", "count"),
                     persen_negatif=("sentimen", lambda s: (s == "negatif").mean()))
                .reset_index())
    masalah_kota = (neg.groupby("kota")["kategori_masalah"]
                    .agg(lambda s: s.value_counts().index[0]).rename("masalah_utama"))
    per_kota = per_kota.merge(masalah_kota, on="kota", how="left").round(3)

    # Contoh keluhan paling banyak disetujui orang lain (like terbanyak)
    contoh = (neg.sort_values("jumlah_like", ascending=False)
              .groupby("kategori_masalah").head(3)[["kategori_masalah", "kota", "teks_llm", "jumlah_like"]])
    return per_kategori, per_kota, contoh


# ---------------------------------------------------------------------------
# 5.2 SKOR PRIORITAS (Decision making yang transparan)
# ---------------------------------------------------------------------------
def hitung_prioritas(per_kategori):
    """skor = 0.6 x (porsi keluhan) + 0.4 x (bobot dampak)."""
    p = per_kategori[per_kategori["kategori_masalah"] != "pujian_layanan"].copy()
    p["bobot_dampak"] = p["kategori_masalah"].map(BOBOT_DAMPAK).fillna(0.3)
    p["skor_prioritas"] = (0.6 * p["persen_dari_keluhan"] + 0.4 * p["bobot_dampak"]).round(3)
    return p.sort_values("skor_prioritas", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5.3 TREN & ANOMALI (bonus "Trend Detection dan Anomaly Detection")
# ---------------------------------------------------------------------------
def deteksi_tren_anomali(df):
    """Hitung keluhan per minggu. Minggu dengan z-score > 2 = anomali
    (jumlah keluhan lebih dari 2 standar deviasi di atas rata-rata)."""
    d = df[df["sentimen"] == "negatif"].copy()
    d["tanggal"] = pd.to_datetime(d["tanggal"], errors="coerce")
    d = d.dropna(subset=["tanggal"])
    if d.empty:
        return pd.DataFrame()
    mingguan = d.set_index("tanggal").resample("W")["id"].count().rename("jumlah_keluhan").reset_index()
    rata, sd = mingguan["jumlah_keluhan"].mean(), mingguan["jumlah_keluhan"].std()
    mingguan["z_score"] = ((mingguan["jumlah_keluhan"] - rata) / (sd if sd else 1)).round(2)
    mingguan["anomali"] = mingguan["z_score"] > 2
    return mingguan


# ---------------------------------------------------------------------------
# 5.4 RAG SEDERHANA (Retrieval-Augmented Generation)
# ---------------------------------------------------------------------------
class PencariPengetahuan:
    """Mencari paragraf fakta yang paling mirip dengan pertanyaan.

    Cara kerja: setiap paragraf diubah menjadi vektor TF-IDF, lalu
    dihitung kemiripan (cosine similarity) dengan kalimat pencarian.
    Paragraf dengan nilai tertinggi diberikan ke LLM sebagai "bukti".
    """

    def __init__(self, path_file):
        teks = open(path_file, encoding="utf-8").read()
        self.paragraf = [p.strip() for p in teks.split("\n\n") if p.strip()]
        self.vectorizer = TfidfVectorizer()
        self.matriks = self.vectorizer.fit_transform(self.paragraf)

    def cari(self, pertanyaan, k=2):
        skor = cosine_similarity(self.vectorizer.transform([pertanyaan]), self.matriks)[0]
        urutan = skor.argsort()[::-1][:k]
        return [self.paragraf[i] for i in urutan if skor[i] > 0]


KATA_PENCARIAN_RAG = {
    "pengangkutan_sampah": "pengangkutan sampah jemput jadwal terangkut tidak terkelola",
    "kebersihan_lingkungan": "sampah tidak terkelola menumpuk TPA open dumping timbulan",
    "fasilitas_sampah": "fasilitas bank sampah TPS3R drop point pemilahan",
    "perilaku_pemilahan": "rumah tangga sumber sampah pemilahan sisa makanan plastik komposisi",
    "hambatan_partisipasi": "partisipasi masyarakat bank sampah jangkauan pengurangan sampah",
    "harga_insentif": "nilai ekonomi daur ulang insentif harga plastik",
    "pelayanan_cs": "layanan pengaduan respon tiket notifikasi",
    "aplikasi_teknis": "layanan pengaduan respon tiket notifikasi aplikasi",
}


def kumpulkan_bukti(prioritas):
    rag = PencariPengetahuan(config.FILE_PENGETAHUAN)
    bukti = []
    for kategori in prioritas["kategori_masalah"].head(3):
        for p in rag.cari(KATA_PENCARIAN_RAG.get(kategori, kategori.replace("_", " "))):
            if p not in bukti:
                bukti.append(p)
    # Fakta umum kondisi sampah nasional selalu disertakan
    for p in rag.cari("timbulan sampah nasional SIPSN tidak terkelola Jakstranas target", k=2):
        if p not in bukti:
            bukti.append(p)
    return bukti


# ---------------------------------------------------------------------------
# 5.5 LLM SEBAGAI ANALIS KEBIJAKAN
# ---------------------------------------------------------------------------
PROMPT_RCA = """Kamu adalah konsultan pengelolaan sampah untuk Dinas Lingkungan Hidup, Pemerintah Daerah,
dan penyedia layanan bank sampah / daur ulang.
Gunakan HANYA data dan fakta di bawah. Jangan mengarang angka. Jika menyebut fakta eksternal,
sebutkan sumbernya seperti tertulis di bagian FAKTA.

DATA HASIL ANALISIS ULASAN ({n_total} ulasan; {n_negatif} negatif):
Prioritas masalah (skor = 0.6 x porsi keluhan + 0.4 x bobot dampak):
{tabel_prioritas}

Ringkasan per kota:
{tabel_kota}

Contoh keluhan nyata:
{contoh_keluhan}

FAKTA EKSTERNAL (hasil retrieval / RAG):
{fakta}

TUGAS — jawab dalam JSON dengan kunci:
"masalah_utama": satu kalimat pernyataan masalah (RCA langkah 1: define the problem),
"rca": list untuk 3 kategori prioritas teratas, tiap item {{"kategori": ..., "lima_why": [5 kalimat "Mengapa ...? Karena ..."], "akar_masalah": ...}},
"fishbone": objek dengan kunci {kategori_fishbone}; nilai tiap kunci = list 2-3 penyebab singkat (maks 6 kata),
"rekomendasi": list 5-7 item {{"prioritas": angka, "untuk": "Penyedia layanan/DLH-Pemda/Masyarakat", "tindakan": ..., "jenis": "korektif" atau "preventif",
   "jangka": "pendek (<3 bln)" / "menengah (3-12 bln)" / "panjang (>1 thn)", "bukti_data": angka dari DATA, "bukti_eksternal": fakta + sumber atau "-",
   "indikator_keberhasilan": ukuran yang bisa dipantau (RCA langkah 6: monitor)}},
"rekomendasi_per_kota": list {{"kota": ..., "fokus": ...}} untuk kota dengan persen negatif tertinggi,
"keterbatasan": 1-2 kalimat keterbatasan analisis ini.
"""


def buat_rca_rekomendasi(df, prioritas, per_kota, contoh, bukti):
    prompt = PROMPT_RCA.format(
        n_total=len(df), n_negatif=int((df["sentimen"] == "negatif").sum()),
        tabel_prioritas=prioritas[["kategori_masalah", "jumlah_keluhan", "persen_dari_keluhan",
                                   "bobot_dampak", "skor_prioritas", "kata_dominan"]].to_string(index=False),
        tabel_kota=per_kota.sort_values("persen_negatif", ascending=False).head(10).to_string(index=False),
        contoh_keluhan="\n".join(f"- [{r.kategori_masalah}, {r.kota}] {r.teks_llm[:200]}"
                                 for r in contoh.itertuples()),
        fakta="\n".join(f"- {b}" for b in bukti),
        kategori_fishbone=json.dumps(KATEGORI_FISHBONE, ensure_ascii=False),
    )
    jawaban = tanya_llm(prompt)
    return baca_json(jawaban, bentuk="objek")


# ---------------------------------------------------------------------------
# 5.6 GAMBAR FISHBONE (Diagram Ishikawa)
# ---------------------------------------------------------------------------
def gambar_fishbone(masalah, penyebab, path_png):
    """Menggambar diagram tulang ikan: 3 tulang di atas, 3 di bawah."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")

    ax.annotate("", xy=(12.6, 4), xytext=(0.5, 4),
                arrowprops=dict(arrowstyle="-|>", lw=3, color="#1f3b57"))         # tulang utama
    ax.text(14.3, 4, "\n".join(_bungkus(masalah, 22)), ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.6", fc="#d64545", ec="none"),
            color="white")                                                         # kepala ikan

    posisi_x = [2.8, 6.6, 10.4]
    for i, kategori in enumerate(KATEGORI_FISHBONE):
        atas = i < 3
        x = posisi_x[i % 3]
        y_ujung = 7.2 if atas else 0.8
        ax.plot([x, x + 1.6], [y_ujung, 4], color="#1f3b57", lw=2)                # tulang cabang
        ax.text(x, y_ujung + (0.35 if atas else -0.35), kategori, ha="center", fontsize=10.5,
                fontweight="bold", color="#1f3b57",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8eef5", ec="#1f3b57"))
        for j, sebab in enumerate((penyebab.get(kategori) or [])[:3]):
            t = (j + 1) / 4
            px = x + 1.6 * t
            py = y_ujung + (4 - y_ujung) * t
            ax.plot([px - 1.3, px], [py, py], color="#7a8ca0", lw=1)
            ax.text(px - 1.35, py, sebab, ha="right", va="center", fontsize=8.5)
    ax.set_title("Fishbone Diagram — Akar Masalah Pengelolaan Sampah", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close(fig)


def _bungkus(teks, lebar):
    kata, baris, sekarang = str(teks).split(), [], ""
    for k in kata:
        if len(sekarang) + len(k) + 1 > lebar:
            baris.append(sekarang); sekarang = k
        else:
            sekarang = f"{sekarang} {k}".strip()
    return baris + [sekarang]


# ---------------------------------------------------------------------------
# FUNGSI UTAMA LANGKAH 5
# ---------------------------------------------------------------------------
def jalankan_rca(df):
    print("\n[LANGKAH 5] Problem analysis, RCA, fishbone, rekomendasi")
    per_kategori, per_kota, contoh = ringkas_data(df)
    prioritas = hitung_prioritas(per_kategori)
    tren = deteksi_tren_anomali(df)
    bukti = kumpulkan_bukti(prioritas)
    print(f"   Prioritas teratas: {prioritas['kategori_masalah'].head(3).tolist()}")
    print(f"   RAG: {len(bukti)} paragraf fakta ditemukan")

    hasil = buat_rca_rekomendasi(df, prioritas, per_kota, contoh, bukti)
    hasil["bukti_rag"] = bukti

    gambar_fishbone(hasil.get("masalah_utama", "Keluhan pengelolaan sampah"),
                    hasil.get("fishbone", {}), os.path.join(config.FOLDER_OUTPUT, "fishbone.png"))

    simpan_csv(prioritas, "prioritas_masalah.csv")
    simpan_csv(per_kota, "ringkasan_per_kota.csv")
    simpan_csv(contoh, "contoh_keluhan.csv")
    if len(tren):
        simpan_csv(tren, "tren_mingguan.csv")
        print(f"   Tren: {int(tren['anomali'].sum())} minggu anomali terdeteksi")
    with open(os.path.join(config.FOLDER_OUTPUT, "rca_rekomendasi.json"), "w") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)
    print("   -> output/rca_rekomendasi.json, output/fishbone.png")
    return hasil, prioritas, per_kota, tren


if __name__ == "__main__":
    jalankan_rca(pd.read_csv(config.FILE_HASIL_LLM))
