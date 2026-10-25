"""
===========================================================================
 LANGKAH 4 — DECISION MAKING ANALYSIS (Evaluasi Keputusan AI)
===========================================================================
 Menjawab pertanyaan dosen dengan BUKTI (angka), bukan hanya opini:

  4.1 Seberapa tepat keputusan Gemini?      -> dibandingkan dengan rating bintang
  4.2 Kapan AI gagal?                        -> uji 25 kasus sulit (sarkasme,
                                                ambigu, campuran, negasi, dialek)
  4.3 Apakah prompt memengaruhi keputusan?   -> prompt SEDERHANA vs DETAIL
  4.4 Apakah Gemini konsisten?               -> teks yang sama dianalisis 3x
  4.5 Apakah confidence bisa dipercaya?      -> akurasi per tingkat confidence
  4.6 (Bonus) Gemini vs LLM lain             -> jika API key pembanding diisi

 Output : folder output/ (file CSV + ringkasan_eksperimen.json)
===========================================================================
"""
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

import config
from langkah3_analisis_llm import analisis_daftar_teks

LABEL = config.LABEL_SENTIMEN


def simpan(df, nama_file):
    df.to_csv(os.path.join(config.FOLDER_OUTPUT, nama_file), index=False)


# ---------------------------------------------------------------------------
# 4.1 EVALUASI TERHADAP RATING BINTANG
# ---------------------------------------------------------------------------
def evaluasi_vs_bintang(df):
    """Rating bintang dipakai sebagai label pembanding (proxy).

    Catatan penting untuk laporan: rating bintang BUKAN kebenaran mutlak.
    Contoh: orang memberi bintang 5 tetapi isinya keluhan (lihat contoh
    Shopee di materi kuliah). Justru perbedaan inilah bahan analisis.
    """
    d = df[df["label_bintang"].isin(LABEL) & df["sentimen"].isin(LABEL)]
    if d.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    akurasi = accuracy_score(d["label_bintang"], d["sentimen"])
    f1 = f1_score(d["label_bintang"], d["sentimen"], average="macro")
    matriks = pd.DataFrame(confusion_matrix(d["label_bintang"], d["sentimen"], labels=LABEL),
                           index=[f"bintang_{l}" for l in LABEL],
                           columns=[f"llm_{l}" for l in LABEL])

    # Contoh perbedaan paling mencolok: bintang tinggi tapi LLM bilang negatif, dst.
    beda = d[d["label_bintang"] != d["sentimen"]]
    beda = beda.sort_values("confidence", ascending=False)
    kolom = ["id", "teks_llm", "rating", "label_bintang", "sentimen", "confidence", "alasan"]
    contoh_beda = beda[kolom].head(30)

    hasil = {"jumlah_dibandingkan": len(d), "akurasi": round(akurasi, 4), "f1_macro": round(f1, 4),
             "jumlah_berbeda": len(beda)}
    print(f"   4.1 Akurasi vs bintang: {akurasi:.2%} | F1-macro: {f1:.3f} (n={len(d)})")
    simpan(matriks.reset_index(names="aktual"), "confusion_matrix.csv")
    simpan(contoh_beda, "contoh_beda_llm_vs_bintang.csv")
    return hasil, matriks, contoh_beda


# ---------------------------------------------------------------------------
# 4.2 UJI KASUS SULIT (bahan "analisis minimal 3 jenis kegagalan AI")
# ---------------------------------------------------------------------------
def uji_kasus_sulit(daftar_penyedia):
    kasus = pd.read_csv(config.FILE_KASUS_SULIT)
    masukan = kasus.rename(columns={"no": "id", "teks": "teks_llm"})[["id", "teks_llm"]]

    semua = []
    for penyedia, versi in daftar_penyedia:
        hasil = analisis_daftar_teks(masukan, versi, penyedia)
        hasil = kasus.merge(hasil, left_on="no", right_on="id", how="left")
        hasil["model_prompt"] = f"{penyedia}/{versi}"
        hasil["benar"] = hasil["sentimen"] == hasil["label_harapan"]
        semua.append(hasil)
    semua = pd.concat(semua, ignore_index=True)

    # Akurasi per jenis kasus, per model/prompt
    per_jenis = semua.pivot_table(index="jenis", columns="model_prompt", values="benar",
                                  aggfunc="mean").round(2)
    salah = semua[~semua["benar"]][["model_prompt", "no", "jenis", "teks", "label_harapan",
                                     "sentimen", "confidence", "alasan"]]
    print("   4.2 Akurasi kasus sulit:",
          semua.groupby("model_prompt")["benar"].mean().round(2).to_dict())
    simpan(semua, "hasil_kasus_sulit.csv")
    simpan(per_jenis.reset_index(), "akurasi_kasus_sulit_per_jenis.csv")
    simpan(salah, "kasus_salah_klasifikasi.csv")
    return semua, per_jenis, salah


# ---------------------------------------------------------------------------
# 4.3 EKSPERIMEN PROMPT: SEDERHANA vs DETAIL
# ---------------------------------------------------------------------------
def ambil_sampel(df, n):
    d = df[df["label_bintang"].isin(LABEL)]
    return d.sample(min(n, len(d)), random_state=config.SEED_ACAK)


def eksperimen_prompt(df):
    sampel = ambil_sampel(df, config.JUMLAH_SAMPEL_EKSPERIMEN_PROMPT)
    a = analisis_daftar_teks(sampel[["id", "teks_llm"]], "sederhana", "gemini")
    b = analisis_daftar_teks(sampel[["id", "teks_llm"]], "detail", "gemini")
    gabung = (sampel[["id", "teks_llm", "label_bintang"]]
              .merge(a[["id", "sentimen", "confidence"]], on="id")
              .merge(b[["id", "sentimen", "confidence", "alasan"]], on="id",
                     suffixes=("_sederhana", "_detail")))

    hasil = {
        "n": len(gabung),
        "akurasi_sederhana": round(accuracy_score(gabung["label_bintang"], gabung["sentimen_sederhana"]), 4),
        "akurasi_detail": round(accuracy_score(gabung["label_bintang"], gabung["sentimen_detail"]), 4),
        "persen_keputusan_sama": round((gabung["sentimen_sederhana"] == gabung["sentimen_detail"]).mean(), 4),
        "rata_confidence_sederhana": round(gabung["confidence_sederhana"].mean(), 3),
        "rata_confidence_detail": round(gabung["confidence_detail"].mean(), 3),
    }
    print(f"   4.3 Prompt sederhana {hasil['akurasi_sederhana']:.2%} vs detail "
          f"{hasil['akurasi_detail']:.2%} | keputusan sama {hasil['persen_keputusan_sama']:.2%}")
    simpan(gabung, "eksperimen_prompt.csv")
    return hasil, gabung


# ---------------------------------------------------------------------------
# 4.4 UJI KONSISTENSI
# ---------------------------------------------------------------------------
def uji_konsistensi(df):
    """Teks yang SAMA dikirim beberapa kali. LLM bersifat probabilistik
    (memilih kata berdasarkan peluang), jadi jawabannya bisa berubah."""
    sampel = ambil_sampel(df, config.JUMLAH_SAMPEL_KONSISTENSI)[["id", "teks_llm"]]
    tabel = sampel.copy()
    for ke in range(1, config.JUMLAH_PENGULANGAN_KONSISTENSI + 1):
        h = analisis_daftar_teks(sampel, "detail", "gemini")
        tabel = tabel.merge(h[["id", "sentimen", "confidence"]].rename(
            columns={"sentimen": f"sentimen_{ke}", "confidence": f"conf_{ke}"}), on="id")

    kolom_s = [c for c in tabel if c.startswith("sentimen_")]
    kolom_c = [c for c in tabel if c.startswith("conf_")]
    tabel["konsisten"] = tabel[kolom_s].nunique(axis=1) == 1
    tabel["selisih_confidence"] = tabel[kolom_c].max(axis=1) - tabel[kolom_c].min(axis=1)

    hasil = {"n": len(tabel), "pengulangan": len(kolom_s),
             "persen_konsisten": round(tabel["konsisten"].mean(), 4),
             "rata_selisih_confidence": round(tabel["selisih_confidence"].mean(), 3)}
    print(f"   4.4 Konsistensi: {hasil['persen_konsisten']:.2%} teks berlabel sama di semua pengulangan")
    simpan(tabel, "uji_konsistensi.csv")
    return hasil, tabel


# ---------------------------------------------------------------------------
# 4.5 ANALISIS CONFIDENCE (confidence reasoning)
# ---------------------------------------------------------------------------
def analisis_confidence(df):
    d = df[df["label_bintang"].isin(LABEL) & df["sentimen"].isin(LABEL)].copy()
    d["tingkat_confidence"] = pd.cut(d["confidence"], [0, 0.6, 0.8, 1.0001],
                                     labels=["rendah (<0.6)", "sedang (0.6-0.8)", "tinggi (>=0.8)"],
                                     right=False)
    d["cocok_bintang"] = d["sentimen"] == d["label_bintang"]
    tabel = d.groupby("tingkat_confidence", observed=False).agg(
        jumlah=("id", "count"), akurasi_vs_bintang=("cocok_bintang", "mean")).round(3)

    penanda = {k: int(df[k].astype(str).str.lower().eq("true").sum())
               for k in ["ambigu", "sarkasme", "sentimen_campuran"] if k in df}
    print("   4.5 Confidence:", tabel["akurasi_vs_bintang"].to_dict(), "| penanda:", penanda)
    simpan(tabel.reset_index(), "akurasi_per_confidence.csv")
    return tabel, penanda


# ---------------------------------------------------------------------------
# 4.6 PERBANDINGAN GEMINI vs MODEL LAIN (BONUS)
# ---------------------------------------------------------------------------
def perbandingan_model(df):
    sampel = ambil_sampel(df, config.JUMLAH_SAMPEL_EKSPERIMEN_PROMPT)
    lain = analisis_daftar_teks(sampel[["id", "teks_llm"]], "detail", "pembanding")
    gabung = sampel[["id", "teks_llm", "label_bintang", "sentimen"]].rename(
        columns={"sentimen": "sentimen_gemini"}).merge(
        lain[["id", "sentimen"]].rename(columns={"sentimen": "sentimen_pembanding"}), on="id")
    valid = gabung[gabung["sentimen_pembanding"].isin(LABEL) & gabung["sentimen_gemini"].isin(LABEL)]
    hasil = {
        "model_pembanding": f"{config.PEMBANDING_NAMA} ({config.PEMBANDING_MODEL})",
        "n": len(valid),
        "akurasi_gemini": round(accuracy_score(valid["label_bintang"], valid["sentimen_gemini"]), 4),
        "akurasi_pembanding": round(accuracy_score(valid["label_bintang"], valid["sentimen_pembanding"]), 4),
        "cohen_kappa_kesepakatan": round(cohen_kappa_score(valid["sentimen_gemini"],
                                                           valid["sentimen_pembanding"]), 3),
    }
    print(f"   4.6 Gemini {hasil['akurasi_gemini']:.2%} vs {config.PEMBANDING_NAMA} "
          f"{hasil['akurasi_pembanding']:.2%} | kappa {hasil['cohen_kappa_kesepakatan']}")
    simpan(gabung, "perbandingan_model.csv")
    return hasil


# ---------------------------------------------------------------------------
# FUNGSI UTAMA LANGKAH 4
# ---------------------------------------------------------------------------
def jalankan_semua_eksperimen(df):
    print("\n[LANGKAH 4] Decision making analysis")
    ada_pembanding = bool(config.PEMBANDING_API_KEY)

    ringkasan = {}
    ringkasan["evaluasi_vs_bintang"], _, _ = evaluasi_vs_bintang(df)

    penyedia = [("gemini", "sederhana"), ("gemini", "detail")]
    if ada_pembanding:
        penyedia.append(("pembanding", "detail"))
    semua, per_jenis, salah = uji_kasus_sulit(penyedia)
    ringkasan["kasus_sulit_akurasi"] = semua.groupby("model_prompt")["benar"].mean().round(3).to_dict()

    ringkasan["eksperimen_prompt"], _ = eksperimen_prompt(df)
    ringkasan["konsistensi"], _ = uji_konsistensi(df)
    tabel_conf, ringkasan["penanda_teks_sulit"] = analisis_confidence(df)
    ringkasan["akurasi_per_confidence"] = tabel_conf["akurasi_vs_bintang"].to_dict()
    ringkasan["perbandingan_model"] = perbandingan_model(df) if ada_pembanding else "dilewati (tanpa API key)"

    with open(os.path.join(config.FOLDER_OUTPUT, "ringkasan_eksperimen.json"), "w") as f:
        json.dump(ringkasan, f, indent=2, ensure_ascii=False, default=str)
    return ringkasan


if __name__ == "__main__":
    jalankan_semua_eksperimen(pd.read_csv(config.FILE_HASIL_LLM))
