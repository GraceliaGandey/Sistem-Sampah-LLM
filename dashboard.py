"""
===========================================================================
 dashboard.py — DASHBOARD INTERAKTIF (Bonus: visualisasi dashboard, XAI,
                peta, demo sistem saat presentasi)
===========================================================================
 Jalankan SETELAH main.py selesai:
     streamlit run dashboard.py
 Browser akan terbuka di http://localhost:8501
===========================================================================
"""
import html
import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

import config

st.set_page_config(page_title="Analisis Pengelolaan Sampah", page_icon="♻️", layout="wide")
WARNA = {"positif": "#2e9e5b", "negatif": "#d64545", "netral": "#9aa3ad"}
OUT = config.FOLDER_OUTPUT


# ---------------------------------------------------------------------------
# Memuat data hasil main.py
# ---------------------------------------------------------------------------
@st.cache_data
def muat_data():
    df = pd.read_csv(config.FILE_HASIL_LLM)
    df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    for kolom in ["ambigu", "sarkasme", "sentimen_campuran"]:
        df[kolom] = df[kolom].astype(str).str.lower().eq("true")
    return df


def muat_json(nama):
    p = os.path.join(OUT, nama)
    return json.load(open(p)) if os.path.exists(p) else {}


def muat_csv(nama):
    p = os.path.join(OUT, nama)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def tandai_kata_kunci(teks, kata_kunci):
    """Explainable AI: memberi warna kuning pada kata yang dipakai LLM untuk memutuskan."""
    hasil = html.escape(str(teks))
    for k in str(kata_kunci).split(","):
        k = k.strip()
        if len(k) > 2:
            hasil = re.sub(re.escape(html.escape(k)), lambda m: f"<mark>{m.group(0)}</mark>",
                           hasil, flags=re.IGNORECASE)
    return hasil


if not os.path.exists(config.FILE_HASIL_LLM):
    st.error("Belum ada hasil. Jalankan dulu: `python main.py`")
    st.stop()

df_semua = muat_data()

# ---------------------------------------------------------------------------
# Sidebar: filter & API key
# ---------------------------------------------------------------------------
st.sidebar.title("♻️ Filter")
kota_pilihan = st.sidebar.multiselect("Kota / aplikasi", sorted(df_semua["kota"].dropna().unique()))
sentimen_pilihan = st.sidebar.multiselect("Sentimen", config.LABEL_SENTIMEN, default=config.LABEL_SENTIMEN)
df = df_semua[df_semua["sentimen"].isin(sentimen_pilihan)]
if kota_pilihan:
    df = df[df["kota"].isin(kota_pilihan)]

st.sidebar.divider()
kunci = st.sidebar.text_input("Gemini API key (untuk tab Coba Langsung)", type="password",
                              value=config.GEMINI_API_KEY)
config.GEMINI_API_KEY = kunci

st.title("Waste Management Sentiment Analysis")
st.caption(f"LLM-Based Social Problem Analysis & Recommendation System · model: {config.MODEL_GEMINI}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Ringkasan", "🗺️ Peta & Wilayah", "🔍 Explainable AI", "🧪 Evaluasi Keputusan AI",
     "🦴 RCA & Rekomendasi", "🤖 Coba Langsung"])

# ---------------------------------------------------------------------------
# TAB 1 — Ringkasan
# ---------------------------------------------------------------------------
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jumlah ulasan", len(df))
    k2.metric("Negatif", f"{(df['sentimen'] == 'negatif').mean():.1%}")
    k3.metric("Positif", f"{(df['sentimen'] == 'positif').mean():.1%}")
    k4.metric("Rata-rata confidence", f"{df['confidence'].mean():.2f}")

    c1, c2 = st.columns(2)
    hitung = df["sentimen"].value_counts().reset_index()
    c1.plotly_chart(px.pie(hitung, names="sentimen", values="count", hole=0.45, color="sentimen",
                           color_discrete_map=WARNA, title="Distribusi sentimen"), use_container_width=True)
    neg = df[df["sentimen"] == "negatif"]["kategori_masalah"].value_counts().reset_index()
    c2.plotly_chart(px.bar(neg, x="count", y="kategori_masalah", orientation="h",
                           title="Kategori masalah (ulasan negatif)", color_discrete_sequence=["#d64545"]),
                    use_container_width=True)

    tren = df[df["sentimen"] == "negatif"].dropna(subset=["tanggal"])
    if len(tren):
        mingguan = tren.set_index("tanggal").resample("W")["id"].count().reset_index()
        st.plotly_chart(px.line(mingguan, x="tanggal", y="id", markers=True,
                                labels={"id": "ulasan negatif"}, title="Tren keluhan mingguan"),
                        use_container_width=True)

    st.subheader("Prioritas masalah")
    st.caption("skor = 0.6 × porsi keluhan + 0.4 × bobot dampak")
    st.dataframe(muat_csv("prioritas_masalah.csv"), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Peta
# ---------------------------------------------------------------------------
with tab2:
    ringkas = (df.groupby("kota").agg(jumlah=("id", "count"),
                                      persen_negatif=("sentimen", lambda s: (s == "negatif").mean()),
                                      masalah_utama=("kategori_masalah", lambda s: s.value_counts().index[0]))
               .reset_index())
    ringkas = ringkas[ringkas["kota"].isin(config.KOORDINAT_KOTA) & ~ringkas["kota"].isin(["Nasional", "Multi-kota"])]
    ringkas["lat"] = ringkas["kota"].map(lambda k: config.KOORDINAT_KOTA[k][0])
    ringkas["lon"] = ringkas["kota"].map(lambda k: config.KOORDINAT_KOTA[k][1])
    if len(ringkas):
        peta = px.scatter_map(ringkas, lat="lat", lon="lon", size="jumlah", color="persen_negatif",
                              color_continuous_scale="RdYlGn_r", range_color=(0, 1), hover_name="kota",
                              hover_data={"masalah_utama": True, "lat": False, "lon": False},
                              zoom=3.8, center={"lat": -2.5, "lon": 118}, height=520,
                              map_style="open-street-map")
        st.plotly_chart(peta, use_container_width=True)
    st.dataframe(ringkas.drop(columns=["lat", "lon"]), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Explainable AI
# ---------------------------------------------------------------------------
with tab3:
    st.write("Setiap keputusan disertai **kata kunci** (ditandai kuning), **alasan**, dan **confidence**.")
    cari = st.text_input("Cari kata dalam ulasan (misal: jemput, harga, numpuk)")
    hanya_sulit = st.checkbox("Tampilkan hanya teks sulit (ambigu / sarkasme / campuran)")
    d = df
    if cari:
        d = d[d["teks_llm"].str.contains(cari, case=False, na=False)]
    if hanya_sulit:
        d = d[d[["ambigu", "sarkasme", "sentimen_campuran"]].any(axis=1)]
    st.caption(f"{len(d)} ulasan · menampilkan 25 pertama")
    for r in d.head(25).itertuples():
        penanda = [n for n, v in [("ambigu", r.ambigu), ("sarkasme", r.sarkasme),
                                  ("campuran", r.sentimen_campuran)] if v]
        st.markdown(
            f"<div style='border-left:5px solid {WARNA.get(r.sentimen, '#888')};padding:6px 12px;margin:8px 0'>"
            f"<b>{r.sentimen.upper()}</b> · {r.kategori_masalah} · confidence {r.confidence:.2f} · "
            f"⭐ {r.rating if pd.notna(r.rating) else '-'} · {r.kota} {' · 🚩 ' + ', '.join(penanda) if penanda else ''}"
            f"<br>{tandai_kata_kunci(r.teks_llm, r.kata_kunci)}"
            f"<br><i>Alasan AI: {html.escape(str(r.alasan))}</i></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 4 — Evaluasi keputusan AI
# ---------------------------------------------------------------------------
with tab4:
    ring = muat_json("ringkasan_eksperimen.json")
    ev, ep, ko = ring.get("evaluasi_vs_bintang", {}), ring.get("eksperimen_prompt", {}), ring.get("konsistensi", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cocok dgn rating bintang", f"{ev.get('akurasi', 0):.1%}")
    m2.metric("F1-macro", f"{ev.get('f1_macro', 0):.3f}")
    m3.metric("Konsistensi", f"{ko.get('persen_konsisten', 0):.1%}")
    m4.metric("Prompt detail vs sederhana", f"{ep.get('akurasi_detail', 0):.1%}",
              f"{(ep.get('akurasi_detail', 0) - ep.get('akurasi_sederhana', 0)) * 100:+.1f} poin")
    c1, c2 = st.columns(2)
    for kolom, gambar in [(c1, "grafik_confusion_matrix.png"), (c2, "grafik_kasus_sulit.png")]:
        if os.path.exists(os.path.join(OUT, gambar)):
            kolom.image(os.path.join(OUT, gambar))
    st.subheader("Kasus yang salah diklasifikasi (kasus uji sulit)")
    st.dataframe(muat_csv("kasus_salah_klasifikasi.csv"), use_container_width=True)
    st.subheader("Akurasi per tingkat confidence")
    st.dataframe(muat_csv("akurasi_per_confidence.csv"))
    st.subheader("Perbandingan model")
    st.write(ring.get("perbandingan_model", "-"))

# ---------------------------------------------------------------------------
# TAB 5 — RCA & rekomendasi
# ---------------------------------------------------------------------------
with tab5:
    rca = muat_json("rca_rekomendasi.json")
    st.info(f"**Pernyataan masalah:** {rca.get('masalah_utama', '-')}")
    for item in rca.get("rca", []):
        with st.expander(f"5 Whys — {item.get('kategori')}"):
            for i, w in enumerate(item.get("lima_why", []), 1):
                st.write(f"{i}. {w}")
            st.success(f"Akar masalah: {item.get('akar_masalah')}")
    if os.path.exists(os.path.join(OUT, "fishbone.png")):
        st.image(os.path.join(OUT, "fishbone.png"))
    st.subheader("Rekomendasi")
    st.dataframe(pd.DataFrame(rca.get("rekomendasi", [])), use_container_width=True)
    st.subheader("Fokus per kota")
    st.dataframe(pd.DataFrame(rca.get("rekomendasi_per_kota", [])), use_container_width=True)
    with st.expander("Fakta eksternal yang dipakai (RAG)"):
        for b in rca.get("bukti_rag", []):
            st.write("•", b)

# ---------------------------------------------------------------------------
# TAB 6 — Coba langsung (demo saat presentasi)
# ---------------------------------------------------------------------------
with tab6:
    from langkah1_pengumpulan_data import ambil_dari_link
    from langkah2_preprocessing import bersihkan_untuk_llm
    from langkah3_analisis_llm import analisis_daftar_teks, analisis_satu_teks

    st.write("Masukkan **kalimat** atau **link** (Google Play / website). Sistem akan mengambil teks "
             "dan meminta Gemini memutuskan sentimen beserta alasannya.")
    masukan = st.text_area("Kalimat atau link", "Aplikasinya keren tapi penjemputan sering telat")
    if st.button("Analisis", type="primary"):
        if not config.GEMINI_API_KEY:
            st.error("Isi Gemini API key di sidebar dulu.")
        elif masukan.strip().startswith("http"):
            with st.spinner("Mengambil teks dari link..."):
                data = pd.DataFrame(ambil_dari_link(masukan.strip(), jumlah=40))
            if data.empty:
                st.warning("Tidak ada teks yang bisa diambil dari link ini.")
            else:
                data["id"] = range(1, len(data) + 1)
                data["teks_llm"] = data["teks"].apply(bersihkan_untuk_llm)
                with st.spinner(f"Gemini menganalisis {len(data)} teks..."):
                    hasil = data.merge(analisis_daftar_teks(data[["id", "teks_llm"]]), on="id")
                st.plotly_chart(px.pie(hasil, names="sentimen", color="sentimen", color_discrete_map=WARNA,
                                       hole=0.45), use_container_width=True)
                st.dataframe(hasil[["teks_llm", "sentimen", "kategori_masalah", "confidence", "alasan"]],
                             use_container_width=True)
        else:
            with st.spinner("Gemini sedang memutuskan..."):
                h = analisis_satu_teks(bersihkan_untuk_llm(masukan))
            st.markdown(f"### {h['sentimen'].upper()}  ·  {h['kategori_masalah']}")
            st.progress(h["confidence"], text=f"confidence {h['confidence']:.2f}")
            st.markdown(tandai_kata_kunci(masukan, h["kata_kunci"]), unsafe_allow_html=True)
            st.write("**Alasan AI:**", h["alasan"])
            st.write({"ambigu": h["ambigu"], "sarkasme": h["sarkasme"], "sentimen_campuran": h["sentimen_campuran"]})
