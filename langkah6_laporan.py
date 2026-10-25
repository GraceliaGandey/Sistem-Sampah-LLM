"""
===========================================================================
 LANGKAH 6 — VISUALISASI, PETA & LAPORAN OTOMATIS
===========================================================================
 Membuat:
   - output/grafik_*.png        : grafik untuk laporan & slide presentasi
   - output/peta_keluhan.html   : peta interaktif keluhan per kota (bonus map)
   - output/laporan_hasil.html  : laporan hasil lengkap. Buka di browser,
                                  lalu Ctrl+P -> "Save as PDF".
===========================================================================
"""
import base64
import html
import json
import os

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

WARNA = {"positif": "#2e9e5b", "negatif": "#d64545", "netral": "#9aa3ad", "tidak_valid": "#444444"}
OUT = config.FOLDER_OUTPUT


def path(nama):
    return os.path.join(OUT, nama)


def baca_csv_jika_ada(nama):
    return pd.read_csv(path(nama)) if os.path.exists(path(nama)) else pd.DataFrame()


# ---------------------------------------------------------------------------
# 6.1 GRAFIK
# ---------------------------------------------------------------------------
def simpan_grafik(fig, nama):
    fig.tight_layout()
    fig.savefig(path(nama), dpi=140)
    plt.close(fig)


def buat_grafik(df):
    # (a) Distribusi sentimen
    hitung = df["sentimen"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(hitung, labels=[f"{k}\n{v} ({v / hitung.sum():.1%})" for k, v in hitung.items()],
           colors=[WARNA.get(k, "#888") for k in hitung.index], startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.set_title("Distribusi Sentimen (keputusan Gemini)")
    simpan_grafik(fig, "grafik_sentimen.png")

    # (b) Kategori masalah pada ulasan negatif
    neg = df[df["sentimen"] == "negatif"]["kategori_masalah"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(neg.index, neg.values, color="#d64545")
    for i, v in enumerate(neg.values):
        ax.text(v, i, f" {v}", va="center")
    ax.set_title("Kategori Masalah pada Ulasan Negatif"); ax.set_xlabel("jumlah ulasan")
    ax.spines[["top", "right"]].set_visible(False)
    simpan_grafik(fig, "grafik_kategori_masalah.png")

    # (c) Sentimen per kota
    tabel = pd.crosstab(df["kota"], df["sentimen"])
    tabel = tabel.loc[tabel.sum(axis=1).sort_values().index].tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    kiri = None
    for label in [l for l in ["negatif", "netral", "positif"] if l in tabel]:
        ax.barh(tabel.index, tabel[label], left=kiri, color=WARNA[label], label=label)
        kiri = tabel[label] if kiri is None else kiri + tabel[label]
    ax.legend(); ax.set_title("Sentimen per Wilayah / Aplikasi"); ax.set_xlabel("jumlah ulasan")
    ax.spines[["top", "right"]].set_visible(False)
    simpan_grafik(fig, "grafik_sentimen_per_kota.png")

    # (d) Histogram confidence
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in config.LABEL_SENTIMEN:
        ax.hist(df[df["sentimen"] == label]["confidence"], bins=10, range=(0, 1), alpha=0.6,
                color=WARNA[label], label=label)
    ax.legend(); ax.set_title("Sebaran Confidence Keputusan LLM"); ax.set_xlabel("confidence")
    simpan_grafik(fig, "grafik_confidence.png")

    # (e) Confusion matrix LLM vs rating bintang
    cm = baca_csv_jika_ada("confusion_matrix.csv")
    if len(cm):
        m = cm.set_index("aktual")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.imshow(m.values, cmap="Blues")
        ax.set_xticks(range(len(m.columns)), m.columns); ax.set_yticks(range(len(m.index)), m.index)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, m.values[i, j], ha="center", va="center",
                        color="white" if m.values[i, j] > m.values.max() / 2 else "black")
        ax.set_title("Confusion Matrix: Bintang vs Gemini")
        simpan_grafik(fig, "grafik_confusion_matrix.png")

    # (f) Akurasi kasus sulit per jenis
    kj = baca_csv_jika_ada("akurasi_kasus_sulit_per_jenis.csv")
    if len(kj):
        kj = kj.set_index("jenis")
        urutan = [c for c in ["gemini/sederhana", "gemini/detail", "pembanding/detail"] if c in kj]
        kj = kj[urutan]                                  # urutan batang: sederhana, detail, pembanding
        fig, ax = plt.subplots(figsize=(8, 4.5))
        kj.plot(kind="bar", ax=ax, color=["#9aa3ad", "#1f6fb2", "#e8a33d"][:kj.shape[1]])
        ax.set_ylim(0, 1.05); ax.set_ylabel("akurasi"); ax.set_xlabel("")
        ax.set_title("Akurasi pada Kasus Sulit per Jenis (prompt sederhana vs detail)")
        ax.tick_params(axis="x", rotation=20); ax.spines[["top", "right"]].set_visible(False)
        simpan_grafik(fig, "grafik_kasus_sulit.png")

    # (g) Tren mingguan & anomali
    tren = baca_csv_jika_ada("tren_mingguan.csv")
    if len(tren):
        tren["tanggal"] = pd.to_datetime(tren["tanggal"])
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(tren["tanggal"], tren["jumlah_keluhan"], color="#1f6fb2", marker="o", ms=3)
        an = tren[tren["anomali"]]
        ax.scatter(an["tanggal"], an["jumlah_keluhan"], color="#d64545", s=60, zorder=3, label="anomali (z>2)")
        ax.legend(); ax.set_title("Tren Keluhan Mingguan"); ax.set_ylabel("ulasan negatif")
        ax.spines[["top", "right"]].set_visible(False)
        simpan_grafik(fig, "grafik_tren.png")


# ---------------------------------------------------------------------------
# 6.2 PETA (folium / OpenStreetMap)
# ---------------------------------------------------------------------------
def buat_peta(df):
    ringkas = (df.groupby("kota").agg(jumlah=("id", "count"),
                                      persen_negatif=("sentimen", lambda s: (s == "negatif").mean()))
               .reset_index())
    peta = folium.Map(location=[-2.5, 118], zoom_start=5, tiles="OpenStreetMap")
    for r in ringkas.itertuples():
        if r.kota not in config.KOORDINAT_KOTA or r.kota in ("Nasional", "Multi-kota"):
            continue
        lat, lon = config.KOORDINAT_KOTA[r.kota]
        warna = "#d64545" if r.persen_negatif >= 0.6 else ("#e8a33d" if r.persen_negatif >= 0.4 else "#2e9e5b")
        folium.CircleMarker(
            location=[lat, lon], radius=6 + min(r.jumlah, 400) ** 0.5, color=warna, fill=True,
            fill_opacity=0.7,
            popup=f"<b>{r.kota}</b><br>{r.jumlah} ulasan<br>{r.persen_negatif:.0%} negatif",
        ).add_to(peta)
    peta.save(path("peta_keluhan.html"))
    return ringkas


# ---------------------------------------------------------------------------
# 6.3 LAPORAN HTML
# ---------------------------------------------------------------------------
def gambar_html(nama, lebar="100%"):
    if not os.path.exists(path(nama)):
        return ""
    data = base64.b64encode(open(path(nama), "rb").read()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{lebar}">'


def tabel_html(df, kolom=None, maks=20):
    if df is None or len(df) == 0:
        return "<p><i>(tidak ada data)</i></p>"
    d = df[kolom] if kolom else df
    return d.head(maks).to_html(index=False, escape=True, border=0, classes="tabel")


def daftar_rekomendasi_html(rek):
    if not rek:
        return "<p><i>(belum ada)</i></p>"
    return tabel_html(pd.DataFrame(rek))


def buat_laporan(df, laporan_pre):
    ring = json.load(open(path("ringkasan_eksperimen.json"))) if os.path.exists(path("ringkasan_eksperimen.json")) else {}
    rca = json.load(open(path("rca_rekomendasi.json"))) if os.path.exists(path("rca_rekomendasi.json")) else {}
    ev = ring.get("evaluasi_vs_bintang", {})
    ep = ring.get("eksperimen_prompt", {})
    ko = ring.get("konsistensi", {})

    rca_html = ""
    for item in rca.get("rca", []):
        why = "".join(f"<li>{html.escape(str(w))}</li>" for w in item.get("lima_why", []))
        rca_html += (f"<h4>{html.escape(str(item.get('kategori', '')))}</h4><ol>{why}</ol>"
                     f"<p><b>Akar masalah:</b> {html.escape(str(item.get('akar_masalah', '')))}</p>")

    contoh_xai = df[df["sentimen"].isin(config.LABEL_SENTIMEN)].sample(
        min(10, len(df)), random_state=1)[["teks_llm", "sentimen", "confidence", "kata_kunci", "alasan"]]

    isi = f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>Laporan Hasil — Analisis Sentimen Pengelolaan Sampah</title>
<style>
 body{{font-family:Georgia,serif;max-width:900px;margin:30px auto;padding:0 16px;color:#1d2a36;line-height:1.5}}
 h1{{color:#1f3b57}} h2{{border-bottom:2px solid #1f3b57;padding-bottom:4px;margin-top:36px;color:#1f3b57}}
 .tabel{{border-collapse:collapse;width:100%;font-size:12px;font-family:Arial,sans-serif;margin:8px 0}}
 .tabel th{{background:#1f3b57;color:white;padding:5px;text-align:left}} .tabel td{{border-bottom:1px solid #ddd;padding:4px;vertical-align:top}}
 .kotak{{display:flex;gap:12px;flex-wrap:wrap}} .kpi{{flex:1;min-width:150px;background:#eef3f8;padding:10px;border-radius:6px}}
 .kpi b{{font-size:22px;display:block}} .catatan{{background:#fff7e6;padding:8px 12px;border-left:4px solid #e8a33d}}
 @media print{{h2{{page-break-before:auto}} img{{page-break-inside:avoid}}}}
</style></head><body>
<h1>AI-Based Waste Management Sentiment and Community Intervention Recommendation System</h1>
<p><b>Sistem:</b> LLM-Based Social Problem Analysis and Recommendation System &nbsp;|&nbsp;
<b>Model:</b> {config.MODEL_GEMINI} &nbsp;|&nbsp; <b>Kelompok:</b> {", ".join(config.ANGGOTA_KELOMPOK)}</p>

<h2>1. Dataset & Preprocessing</h2>
<div class="kotak">
 <div class="kpi"><b>{laporan_pre.get('data_awal', len(df))}</b>data mentah</div>
 <div class="kpi"><b>{len(df)}</b>data bersih (dianalisis)</div>
 <div class="kpi"><b>{df['kota'].nunique()}</b>kota / aplikasi</div>
 <div class="kpi"><b>{laporan_pre.get('teks_pendek_<=2_kata', '-')}</b>teks sangat pendek (≤2 kata)</div>
</div>
{tabel_html(pd.DataFrame([laporan_pre]))}

<h2>2. Hasil Sentiment Analysis & Problem Analysis</h2>
{gambar_html("grafik_sentimen.png", "48%")}{gambar_html("grafik_confidence.png", "48%")}
{gambar_html("grafik_kategori_masalah.png", "80%")}
{gambar_html("grafik_sentimen_per_kota.png", "80%")}
<p>Peta interaktif: buka file <code>output/peta_keluhan.html</code>.</p>
<h3>Prioritas masalah (skor = 0.6 × porsi keluhan + 0.4 × bobot dampak)</h3>
{tabel_html(baca_csv_jika_ada("prioritas_masalah.csv"))}
{gambar_html("grafik_tren.png", "90%")}

<h2>3. Decision Making Analysis</h2>
<div class="kotak">
 <div class="kpi"><b>{ev.get('akurasi', 0):.1%}</b>kecocokan dengan rating bintang</div>
 <div class="kpi"><b>{ev.get('f1_macro', 0):.3f}</b>F1-macro</div>
 <div class="kpi"><b>{ko.get('persen_konsisten', 0):.1%}</b>konsistensi ({ko.get('pengulangan', '-')}x ulang)</div>
 <div class="kpi"><b>{ep.get('akurasi_sederhana', 0):.1%} → {ep.get('akurasi_detail', 0):.1%}</b>prompt sederhana → detail</div>
</div>
<p class="catatan">Rating bintang hanya label pembanding, bukan kebenaran mutlak: pengguna sering memberi bintang yang tidak sesuai isi ulasan.</p>
{gambar_html("grafik_confusion_matrix.png", "55%")}
<h3>Akurasi per tingkat confidence</h3>{tabel_html(baca_csv_jika_ada("akurasi_per_confidence.csv"))}
<h3>Eksperimen prompt</h3>{tabel_html(pd.DataFrame([ep]))}
<h3>Perbandingan model</h3><p>{html.escape(str(ring.get('perbandingan_model', '-')))}</p>
<h3>Contoh keputusan + penjelasan AI (Explainable AI)</h3>{tabel_html(contoh_xai)}

<h2>4. Analisis Kegagalan AI (sarkasme, ambigu, sentimen campuran)</h2>
{gambar_html("grafik_kasus_sulit.png", "85%")}
<h3>Kasus uji yang salah diklasifikasi</h3>
{tabel_html(baca_csv_jika_ada("kasus_salah_klasifikasi.csv"), maks=40)}
<h3>Contoh ulasan nyata di mana Gemini berbeda dengan rating bintang</h3>
{tabel_html(baca_csv_jika_ada("contoh_beda_llm_vs_bintang.csv"), maks=15)}

<h2>5. Root Cause Analysis & Fishbone</h2>
<p><b>Pernyataan masalah:</b> {html.escape(str(rca.get('masalah_utama', '-')))}</p>
{rca_html}
{gambar_html("fishbone.png")}

<h2>6. Rekomendasi (Decision Making untuk DLH / Pemda / Penyedia Layanan)</h2>
{daftar_rekomendasi_html(rca.get('rekomendasi'))}
<h3>Fokus per kota</h3>{daftar_rekomendasi_html(rca.get('rekomendasi_per_kota'))}
<h3>Fakta eksternal yang dipakai (RAG)</h3>
<ul>{"".join(f"<li>{html.escape(b)}</li>" for b in rca.get('bukti_rag', []))}</ul>
<p class="catatan"><b>Keterbatasan:</b> {html.escape(str(rca.get('keterbatasan', '-')))}</p>
</body></html>"""
    with open(path("laporan_hasil.html"), "w", encoding="utf-8") as f:
        f.write(isi)


# ---------------------------------------------------------------------------
# FUNGSI UTAMA LANGKAH 6
# ---------------------------------------------------------------------------
def jalankan_laporan(df, laporan_pre=None):
    print("\n[LANGKAH 6] Visualisasi, peta, laporan")
    buat_grafik(df)
    buat_peta(df)
    buat_laporan(df, laporan_pre or {})
    print("   -> output/laporan_hasil.html (buka di browser, Ctrl+P untuk PDF)")
    print("   -> output/peta_keluhan.html")


if __name__ == "__main__":
    jalankan_laporan(pd.read_csv(config.FILE_HASIL_LLM))
