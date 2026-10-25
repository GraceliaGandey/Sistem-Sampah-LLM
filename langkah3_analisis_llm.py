"""
===========================================================================
 LANGKAH 3 — ANALISIS SENTIMEN & MASALAH DENGAN LLM (Gemini)
===========================================================================
 Inilah inti "AI Decision Making" pada tugas ini.
 Untuk setiap ulasan, LLM MEMUTUSKAN:
   1. sentimen          : positif / negatif / netral
   2. kategori_masalah  : pengangkutan_sampah, kebersihan_lingkungan, ...
   3. confidence        : seberapa yakin (0.0 - 1.0)
   4. kata_kunci        : kata yang paling memengaruhi keputusan   } Explainable
   5. alasan            : penjelasan singkat keputusan             } AI (XAI)
   6. ambigu / sarkasme / sentimen_campuran : penanda teks sulit

 Cara kerja:
   ulasan (100 sekaligus) --> PROMPT --> Gemini API --> jawaban JSON --> tabel

 Output : data/hasil_llm.csv
===========================================================================
"""
import json
import os
import re
import time

import pandas as pd

import config

# ---------------------------------------------------------------------------
# A. PROMPT  (dua versi, untuk "Prompt engineering eksperimen")
# ---------------------------------------------------------------------------
# Versi A: SEDERHANA (zero-shot, tanpa definisi). Dipakai sebagai pembanding.
PROMPT_SEDERHANA = """Tentukan sentimen setiap ulasan berikut (positif, negatif, atau netral)
dan kategori masalahnya ({kategori}).
Jawab HANYA dalam JSON list: [{{"id": ..., "sentimen": ..., "kategori_masalah": ..., "confidence": 0.0-1.0}}]

Ulasan:
{daftar_ulasan}
"""

# Versi B: DETAIL (peran + konteks + definisi + aturan teks sulit + contoh/few-shot).
# Ini prompt UTAMA yang dipakai untuk seluruh dataset.
PROMPT_DETAIL = """Kamu adalah analis opini publik untuk layanan pengelolaan sampah di Indonesia
(aplikasi bank sampah digital, jemput sampah, dan daur ulang). Tugasmu menilai ulasan pengguna dari
Google Play. Ulasan sering memakai bahasa gaul, singkatan, bahasa daerah, atau campuran bahasa Inggris.

DEFINISI SENTIMEN (nilai sikap penulis terhadap LAYANAN PENGELOLAAN SAMPAH / aplikasinya):
- "positif": puas, memuji, berterima kasih, merasa terbantu atau termotivasi memilah sampah.
- "negatif": kecewa, mengeluh, marah, melaporkan masalah, atau menyindir.
- "netral" : pertanyaan, informasi, saran tanpa emosi jelas, atau teks tidak bermakna.

ATURAN TEKS SULIT:
1. Sarkasme: pujian yang sebenarnya menyindir (contoh: "mantap, sampah 3 minggu belum dijemput") -> negatif, sarkasme=true.
2. Sentimen campuran: ada pujian DAN keluhan -> pilih sentimen yang menjadi POKOK pesan
   (biasanya keluhan yang meminta tindakan), lalu sentimen_campuran=true.
3. Ambigu: makna bisa ditafsirkan lebih dari satu -> pilih tafsiran paling mungkin, ambigu=true,
   dan turunkan confidence.
4. Rating bintang TIDAK diberikan kepadamu; nilai hanya dari isi teks.

KATEGORI MASALAH (pilih satu yang paling utama):
- pengangkutan_sampah  : penjemputan telat/tidak datang, jadwal tidak jelas, kurir/petugas angkut
- kebersihan_lingkungan: sampah menumpuk, bau, TPS penuh, buang sampah sembarangan
- fasilitas_sampah     : drop point / bank sampah jauh atau kurang, tempat sampah tidak ada
- perilaku_pemilahan   : cara memilah, edukasi, kebiasaan masyarakat membuang sampah
- hambatan_partisipasi : minimal berat setor, wilayah belum terjangkau, syarat rumit
- harga_insentif       : harga per kg rendah, poin/saldo, pencairan dana
- pelayanan_cs         : customer service lambat, komplain diabaikan
- aplikasi_teknis      : aplikasi error, login gagal, fitur tidak berjalan
- pujian_layanan       : apresiasi tanpa keluhan
- lainnya              : tidak termasuk kategori di atas

CONFIDENCE: 0.9-1.0 = sangat jelas; 0.6-0.8 = cukup jelas; < 0.6 = ragu/ambigu.

CONTOH:
Ulasan: "Senang banget sekarang sampah plastik di rumah bisa dijemput dan jadi saldo"
-> {{"sentimen": "positif", "kategori_masalah": "pujian_layanan", "confidence": 0.95,
    "kata_kunci": ["senang banget", "dijemput", "jadi saldo"], "alasan": "Pengguna puas karena sampah dijemput dan bernilai uang.",
    "ambigu": false, "sarkasme": false, "sentimen_campuran": false}}
Ulasan: "Fiturnya lengkap, tapi wilayah saya belum bisa request jemput padahal sudah daftar"
-> {{"sentimen": "negatif", "kategori_masalah": "hambatan_partisipasi", "confidence": 0.8,
    "kata_kunci": ["belum bisa request jemput", "padahal sudah daftar"],
    "alasan": "Ada pujian untuk fitur, tetapi pokok pesan adalah wilayah yang belum terjangkau layanan.",
    "ambigu": false, "sarkasme": false, "sentimen_campuran": true}}

Analisis SEMUA ulasan di bawah. Jawab HANYA dengan JSON list, satu objek per ulasan, dengan kunci:
"id", "sentimen", "kategori_masalah", "confidence", "kata_kunci" (list, maks 3), "alasan" (1 kalimat
bahasa Indonesia), "ambigu", "sarkasme", "sentimen_campuran".

Ulasan:
{daftar_ulasan}
"""

DAFTAR_PROMPT = {"sederhana": PROMPT_SEDERHANA, "detail": PROMPT_DETAIL}


# ---------------------------------------------------------------------------
# B. MENGHUBUNGI LLM
# ---------------------------------------------------------------------------
_client_gemini = None
_client_pembanding = None


def tanya_llm(prompt, penyedia="gemini", suhu=None):
    """Mengirim prompt ke LLM dan mengembalikan jawaban berupa teks.

    penyedia = "gemini"     -> Google Gemini (model utama)
    penyedia = "pembanding" -> model lain lewat API OpenAI-compatible (Groq/DeepSeek)
    suhu (temperature)      -> None = pakai bawaan model. Makin tinggi makin "kreatif".
    """
    global _client_gemini, _client_pembanding

    for percobaan in range(1, config.MAKS_ULANG + 1):
        try:
            if penyedia == "gemini":
                from google import genai
                from google.genai import types
                if _client_gemini is None:
                    if not config.GEMINI_API_KEY:
                        raise RuntimeError("GEMINI_API_KEY belum diisi (lihat config.py)")
                    _client_gemini = genai.Client(api_key=config.GEMINI_API_KEY)
                respon = _client_gemini.models.generate_content(
                    model=config.MODEL_GEMINI,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",  # paksa jawaban JSON
                        temperature=suhu,
                        # berpikir singkat saja = jawaban jauh lebih cepat
                        thinking_config=(types.ThinkingConfig(thinking_level=config.TINGKAT_BERPIKIR)
                                         if config.TINGKAT_BERPIKIR else None),
                    ),
                )
                return respon.text

            else:  # model pembanding
                from openai import OpenAI
                if _client_pembanding is None:
                    _client_pembanding = OpenAI(api_key=config.PEMBANDING_API_KEY,
                                                base_url=config.PEMBANDING_BASE_URL)
                respon = _client_pembanding.chat.completions.create(
                    model=config.PEMBANDING_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=suhu if suhu is not None else 0.7,
                )
                return respon.choices[0].message.content

        except RuntimeError:
            raise
        except Exception as error:
            # Kalau model tidak mendukung pengaturan "berpikir", matikan lalu coba lagi
            if "thinking" in str(error).lower() and config.TINGKAT_BERPIKIR:
                print("   ! Model tidak mendukung thinking_level -> dimatikan")
                config.TINGKAT_BERPIKIR = None
                continue
            # Biasanya error 429 (kuota/limit) atau 503 (server sibuk) -> tunggu lalu ulang
            tunggu = 20 * percobaan
            print(f"   ! Error LLM (percobaan {percobaan}): {str(error)[:120]} -> tunggu {tunggu} dtk")
            time.sleep(tunggu)
    raise RuntimeError("LLM gagal menjawab setelah beberapa kali percobaan")


def baca_json(teks_jawaban, bentuk="list"):
    """Mengubah jawaban teks LLM menjadi data Python.

    bentuk="list"  -> hasil berupa list (analisis sentimen per ulasan)
    bentuk="objek" -> hasil berupa dict  (RCA & rekomendasi)
    Kadang LLM membungkus JSON dengan ```json ... ``` atau kalimat lain,
    jadi kita bersihkan dulu, lalu cari bagian [ ... ] atau { ... }.
    """
    teks = teks_jawaban.strip()
    teks = re.sub(r"^```(json)?|```$", "", teks, flags=re.MULTILINE).strip()
    try:
        data = json.loads(teks)
    except json.JSONDecodeError:
        pola = r"\[.*\]" if bentuk == "list" else r"\{.*\}"
        cocok = re.search(pola, teks, flags=re.DOTALL)
        data = json.loads(cocok.group(0)) if cocok else ([] if bentuk == "list" else {})

    if bentuk == "objek":
        return data[0] if isinstance(data, list) and data else (data or {})
    if isinstance(data, dict):          # kalau LLM menjawab {"hasil": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    return data


def rapikan_hasil(item):
    """Memastikan setiap jawaban LLM punya format yang benar (validasi)."""
    sinonim = {"positive": "positif", "negative": "negatif", "neutral": "netral"}
    sentimen = str(item.get("sentimen", "")).lower().strip()
    sentimen = sinonim.get(sentimen, sentimen)
    if sentimen not in config.LABEL_SENTIMEN:
        sentimen = "tidak_valid"

    kategori = str(item.get("kategori_masalah", "lainnya")).lower().strip()
    if kategori not in config.KATEGORI_MASALAH:
        kategori = "lainnya"

    try:
        confidence = min(max(float(item.get("confidence", 0.5)), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.5

    kata_kunci = item.get("kata_kunci", [])
    if isinstance(kata_kunci, list):
        kata_kunci = ", ".join(map(str, kata_kunci))

    try:
        id_teks = int(item.get("id", -1))
    except (TypeError, ValueError):
        id_teks = -1

    return {
        "id": id_teks,
        "sentimen": sentimen,
        "kategori_masalah": kategori,
        "confidence": confidence,
        "kata_kunci": kata_kunci,
        "alasan": item.get("alasan", ""),
        "ambigu": bool(item.get("ambigu", False)),
        "sarkasme": bool(item.get("sarkasme", False)),
        "sentimen_campuran": bool(item.get("sentimen_campuran", False)),
    }


# ---------------------------------------------------------------------------
# C. ANALISIS PER BATCH
# ---------------------------------------------------------------------------
def analisis_batch(daftar_id_teks, versi_prompt="detail", penyedia="gemini", suhu=None):
    """Menganalisis sekumpulan (id, teks) dalam SATU panggilan LLM.

    Mengirim banyak ulasan sekaligus jauh lebih hemat kuota daripada 1-per-1.
    """
    daftar_ulasan = "\n".join(f'- id={i}: "{t}"' for i, t in daftar_id_teks)
    prompt = DAFTAR_PROMPT[versi_prompt].format(
        daftar_ulasan=daftar_ulasan, kategori=", ".join(config.KATEGORI_MASALAH))

    jawaban = tanya_llm(prompt, penyedia=penyedia, suhu=suhu)
    try:
        hasil = [rapikan_hasil(x) for x in baca_json(jawaban)]
    except Exception as error:
        print(f"   ! JSON tidak terbaca: {error}")
        hasil = []

    # Ulasan yang tidak dijawab LLM ditandai "tidak_valid" (dihitung sebagai kegagalan)
    id_terjawab = {h["id"] for h in hasil}
    for i, _ in daftar_id_teks:
        if i not in id_terjawab:
            hasil.append(rapikan_hasil({"id": i, "sentimen": "tidak_valid",
                                        "alasan": "LLM tidak memberi jawaban"}))
    return hasil


def analisis_daftar_teks(df_teks, versi_prompt="detail", penyedia="gemini", suhu=None,
                         file_checkpoint=None):
    """Menganalisis seluruh DataFrame (kolom: id, teks_llm) batch demi batch.

    file_checkpoint: hasil disimpan setiap batch. Kalau program terhenti
    (misal kuota habis), jalankan ulang -> lanjut dari yang belum.
    """
    sudah = pd.DataFrame()
    if file_checkpoint and os.path.exists(file_checkpoint):
        sudah = pd.read_csv(file_checkpoint)
        print(f"   Melanjutkan checkpoint: {len(sudah)} teks sudah dianalisis")
    id_sudah = set(sudah["id"]) if len(sudah) else set()

    sisa = df_teks[~df_teks["id"].isin(id_sudah)]
    pasangan = list(zip(sisa["id"], sisa["teks_llm"]))
    hasil_semua = sudah.to_dict("records")

    for mulai in range(0, len(pasangan), config.UKURAN_BATCH):
        batch = pasangan[mulai:mulai + config.UKURAN_BATCH]
        hasil_semua.extend(analisis_batch(batch, versi_prompt, penyedia, suhu))
        print(f"   [{penyedia}/{versi_prompt}] {min(mulai + len(batch), len(pasangan))}/{len(pasangan)}")
        if file_checkpoint:
            pd.DataFrame(hasil_semua).to_csv(file_checkpoint, index=False)
        time.sleep(config.JEDA_DETIK)

    return pd.DataFrame(hasil_semua)


def analisis_satu_teks(teks, penyedia="gemini"):
    """Untuk demo langsung di dashboard: analisis satu kalimat."""
    return analisis_batch([(1, teks)], "detail", penyedia)[0]


# ---------------------------------------------------------------------------
# FUNGSI UTAMA LANGKAH 3
# ---------------------------------------------------------------------------
def pilih_sampel(df, file_checkpoint):
    """Membatasi jumlah teks sesuai config.MAKS_DATA_ANALISIS.

    Teks yang SUDAH dianalisis di checkpoint selalu diikutkan (supaya kuota
    yang sudah terpakai tidak sia-sia), sisanya diambil acak dari data lain.
    """
    batas = config.MAKS_DATA_ANALISIS
    if not batas or len(df) <= batas:
        return df
    id_sudah = set(pd.read_csv(file_checkpoint)["id"]) if os.path.exists(file_checkpoint) else set()
    sudah = df[df["id"].isin(id_sudah)]
    sisa = df[~df["id"].isin(id_sudah)]
    tambahan = sisa.sample(max(batas - len(sudah), 0), random_state=config.SEED_ACAK)
    print(f"   Data bersih {len(df)} teks -> dianalisis {batas} teks (sampel acak)")
    return pd.concat([sudah, tambahan]).sort_values("id")


def analisis_dataset(df):
    file_checkpoint = os.path.join(config.FOLDER_DATA, "_checkpoint_llm.csv")
    df = pilih_sampel(df, file_checkpoint)
    print(f"\n[LANGKAH 3] Analisis LLM ({config.MODEL_GEMINI}) untuk {len(df)} teks")
    hasil = analisis_daftar_teks(df[["id", "teks_llm"]], "detail", "gemini",
                                 file_checkpoint=file_checkpoint)
    hasil = hasil.drop_duplicates("id", keep="last")
    df_final = df.merge(hasil, on="id", how="left")
    for kolom in ["ambigu", "sarkasme", "sentimen_campuran"]:     # pastikan True/False
        df_final[kolom] = df_final[kolom].astype(str).str.lower().eq("true")
    df_final.to_csv(config.FILE_HASIL_LLM, index=False)

    print("   Distribusi sentimen:", df_final["sentimen"].value_counts().to_dict())
    print(f"   -> {config.FILE_HASIL_LLM}")
    return df_final


if __name__ == "__main__":
    analisis_dataset(pd.read_csv(config.FILE_DATASET_BERSIH))
