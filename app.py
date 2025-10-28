# app.py
# =============================================================================
# Streamlit Chat - Telesales Ruangguru (Role-Play)
# Kompatibel SDK baru (google-genai) dan SDK lama (google.generativeai)
# Persona = Orang Tua/Murid (non-sales), opener oleh assistant, avatar dibedakan,
# tanpa overwrite key widget, tanpa duplikasi render
# =============================================================================
import os
import json
import time
import re
import random
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

import streamlit as st

# === A0: Page config + CSS ====================================================
st.set_page_config(page_title="RG Telesales - Role-Play Chat", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 2.25rem; padding-bottom: 1rem; max-width: 980px;}
.stChatMessage {gap: .25rem;}
html, body {scroll-behavior: smooth;}
@media (max-width: 600px){
  .block-container {padding-left: .6rem; padding-right: .6rem;}
}
/* Sidebar desktop: rapatkan kolom tombol */
@media (min-width: 992px){
  section[data-testid="stSidebar"] div[data-testid="column"]{ padding-right:.25rem !important; }
  section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]{ gap:.25rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# === A1: Deteksi SDK ==========================================================
SDK = "new"  # "new" = google-genai, "legacy" = google.generativeai
try:
    from google import genai as genai_new
    from google.genai import types as types_new
except Exception:
    SDK = "legacy"
    import google.generativeai as genai_legacy
    types_new = None  # tidak dipakai di jalur legacy

# === A2: API key resolver =====================================================
DEMO_API_KEY = "AIzaSyCi19OsrR1lsoN7qs2EU5U4zP-8j_1eHh4"

def _resolve_api_key() -> str:
    try:
        for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GENAI_API_KEY"):
            if k in st.secrets and str(st.secrets[k]).strip():
                return str(st.secrets[k]).strip()
    except Exception:
        pass
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GENAI_API_KEY"):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    return DEMO_API_KEY

API_KEY = _resolve_api_key()

# Model names tanpa prefix "models/" untuk kompatibilitas lintas SDK
MODEL_PRIMARY = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_FALLBACKS = [MODEL_PRIMARY, "gemini-2.0-flash", "gemini-1.5-flash"]

# === A3: Client/Model init ====================================================
@st.cache_resource(show_spinner=False)
def _init_client_or_none():
    if SDK == "new":
        if not API_KEY:
            raise RuntimeError("GOOGLE_API_KEY tidak ditemukan di st.secrets atau environment.")
        return genai_new.Client(api_key=API_KEY)
    else:
        if not API_KEY:
            raise RuntimeError("API key tidak ditemukan untuk SDK lama (google.generativeai).")
        genai_legacy.configure(api_key=API_KEY)
        return None  # jalur legacy tidak memakai objek client

client = _init_client_or_none()

# === A3c: Apply pending load before any widgets ===============================
if st.session_state.get("pending_load"):
    data = st.session_state.pop("pending_load")
    st.session_state.messages = data.get("messages", [])
    st.session_state.aud = data.get("audience", st.session_state.get("aud", "Orang Tua"))
    st.session_state.seg = data.get("segment", st.session_state.get("seg", "SMP"))
    st.session_state.bot_persona = st.session_state.aud
    st.session_state.convo_id = data.get("convo_id")
    st.session_state.convo_title = data.get("title") or ""
    st.session_state.intent = None
    st.session_state.suppress_next_reply = True

# === A3b: Storage (SQLite) ====================================================
DB_PATH = Path(__file__).with_name("telesales_history.sqlite")

@st.cache_resource(show_spinner=False)
def _get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS convo(
            id TEXT PRIMARY KEY,
            title TEXT,
            audience TEXT,
            segment TEXT,
            created_at TEXT,
            updated_at TEXT,
            messages_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_convo_updated ON convo(updated_at DESC)")
    return conn

def _prune_internal_msgs(msgs: List[Dict]) -> List[Dict]:
    intr = set(st.session_state.get("internal_triggers", []))
    return [m for m in msgs if m.get("content") not in intr]

def _derive_title(msgs: List[Dict]) -> str:
    for m in msgs:
        if m.get("role") == "user":
            t = (m.get("content") or "").strip().splitlines()[0]
            return (t[:80] or "Percakapan")
    return f"Chat {datetime.now():%Y-%m-%d %H:%M}"

def save_current_convo():
    conn = _get_db()
    msgs = _prune_internal_msgs(st.session_state.get("messages", []))
    if not msgs:
        return None
    convo_id = st.session_state.get("convo_id") or uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    row = conn.execute("SELECT created_at FROM convo WHERE id=?", (convo_id,)).fetchone()
    created_at = row[0] if row else now
    title = st.session_state.get("convo_title") or _derive_title(msgs)
    conn.execute(
        "INSERT INTO convo(id,title,audience,segment,created_at,updated_at,messages_json) "
        "VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "title=excluded.title,audience=excluded.audience,segment=excluded.segment,updated_at=excluded.updated_at,messages_json=excluded.messages_json",
        (
            convo_id,
            title,
            st.session_state.get("aud", ""),
            st.session_state.get("seg", ""),
            created_at,
            now,
            json.dumps(msgs, ensure_ascii=False),
        ),
    )
    conn.commit()
    st.session_state.convo_id = convo_id
    st.session_state.convo_title = title
    return convo_id

def list_convos() -> List[Dict]:
    conn = _get_db()
    rows = conn.execute(
        "SELECT id,title,audience,segment,created_at,updated_at FROM convo ORDER BY updated_at DESC"
    ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "audience": r[2],
            "segment": r[3],
            "created_at": r[4],
            "updated_at": r[5],
        }
        for r in rows
    ]

def load_convo(convo_id: str):
    conn = _get_db()
    row = conn.execute(
        "SELECT messages_json,audience,segment,title FROM convo WHERE id=?", (convo_id,)
    ).fetchone()
    if not row:
        return
    msgs = json.loads(row[0]) if row[0] else []
    # Tunda pengisian widget-bound keys; terapkan sebelum widget dibuat pada run berikutnya
    st.session_state.pending_load = {
        "messages": msgs,
        "audience": row[1] or st.session_state.get("aud", "Orang Tua"),
        "segment": row[2] or st.session_state.get("seg", "SMP"),
        "title": row[3] or _derive_title(msgs),
        "convo_id": convo_id,
    }
    st.rerun()

def delete_convo(convo_id: str):
    conn = _get_db()
    conn.execute("DELETE FROM convo WHERE id=?", (convo_id,))
    conn.commit()
    if st.session_state.get("convo_id") == convo_id:
        for k in ["convo_id", "convo_title"]:
            st.session_state.pop(k, None)

# === A4: Data katalog (UI saja; tidak dimasukkan ke prompt) ==================
CATALOG = {
    "SD": [
        {"kode": "RB-SD-LIVE", "nama": "RuangBelajar Live SD", "fitur": ["Live interaktif", "Bank soal SD", "Rekaman kelas", "Kuis adaptif"], "cocok": ["kelas 4-6", "butuh latihan rutin"]},
        {"kode": "RB-SD-VIDEO", "nama": "RuangBelajar Video SD", "fitur": ["Video konsep dasar", "Ringkasan materi", "Latihan ringan"], "cocok": ["kelas 1-3", "butuh penguatan konsep"]},
        {"kode": "RO-PRIV", "nama": "RuangOrtu Insight", "fitur": ["Report progres", "Rencana belajar mingguan"], "cocok": ["orang tua ingin pantau"]}
    ],
    "SMP": [
        {"kode": "RB-SMP-LIVE", "nama": "RuangBelajar Live SMP", "fitur": ["Live intensif", "Tryout mingguan", "Pembahasan detail"], "cocok": ["kelas 9", "target naik nilai rapor"]},
        {"kode": "RB-SMP-PAKET", "nama": "Paket Latihan SMP", "fitur": ["Bank soal AKM", "Remedial adaptif"], "cocok": ["kelas 7-9", "butuh drilling"]}
    ],
    "SMA": [
        {"kode": "RB-SMA-LIVE", "nama": "RuangBelajar Live SMA", "fitur": ["Live STEM dan Soshum", "Jadwal intensif"], "cocok": ["kelas 10-12", "persiapan ujian sekolah"]},
        {"kode": "RB-UTBK", "nama": "Paket UTBK", "fitur": ["Tryout UTBK", "Pembahasan SKD", "Strategi waktu"], "cocok": ["kelas 12", "fokus kampus tujuan"]}
    ]
}

# === A4b: Pool skenario opener (segment-aware) ===============================
OPENER_POOL: Dict[str, Dict[str, List[str]]] = {
    "Murid": {
        "SD": [
            "Perkalian dan pembagian masih sering salah; PR Matematika terasa berat.",
            "Sulit memahami bacaan panjang; suka kebingungan cari ide pokok.",
            "IPA dasar tentang tumbuhan dan hewan membingungkan saat ulangan.",
            "Gampang terdistraksi main gim saat jam belajar; susah fokus 20 menit penuh.",
            "Kesulitan menulis rapi dan cepat saat dikte; ketinggalan materi.",
        ],
        "SMP": [
            "Nilai Fisika tentang gaya dan gerak turun; bingung rumus dan satuan.",
            "Aljabar dan persamaan linear bikin mentok; salah di langkah awal.",
            "Bahasa Inggris reading panjang bikin kewalahan; kosakata kurang.",
            "Sering kehabisan waktu karena ekskul; tugas menumpuk jelang ulangan.",
            "Kimia pengantar zat dan perubahan wujud masih rancu.",
        ],
        "SMA": [
            "Trigonometri dan limit membingungkan; nilai kuis turun.",
            "UTBK makin dekat; sulit konsisten belajar setiap hari.",
            "Fisika kinematika sering salah di konversi satuan.",
            "Kimia stoikiometri panjang; bingung analisis mol dan massa.",
            "Ekonomi mikro: elastisitas dan kurva permintaan-penawaran masih salah konsep.",
        ],
    },
    "Orang Tua": {
        "SD": [
            "Anak mudah terdistraksi HP saat belajar; PR sering ditunda.",
            "Membaca pemahaman masih lemah; perlu latihan bertahap.",
            "Sulit duduk fokus lebih dari 15 menit; butuh pola belajar singkat.",
            "Ingin cara memantau progres tanpa harus mendampingi terus-menerus.",
        ],
        "SMP": [
            "Nilai Matematika menurun dua bulan terakhir; remedial sering tidak tuntas.",
            "Anak malu bertanya di kelas; konsep IPA kurang kuat.",
            "Jadwal ekskul padat; tugas dan ulangan sering berbenturan.",
            "Butuh kebiasaan belajar teratur tanpa harus dimarahi.",
        ],
        "SMA": [
            "Persiapan UTBK belum terarah; anak sulit konsisten.",
            "Bingung pemilihan jurusan; perlu arahan fokus mata pelajaran.",
            "Belajar mandiri tapi cepat burnout; perlu ritme yang sehat.",
            "Orangtua ingin laporan progres yang ringkas dan objektif.",
        ],
    },
}

def _sample_scenario(audience: str, segment: str) -> str:
    pool_aud = OPENER_POOL.get(audience, {})
    pool_seg: List[str] = []
    if isinstance(pool_aud, dict):
        pool_seg = pool_aud.get(segment, [])
    if not pool_seg and isinstance(pool_aud, dict):
        merged: List[str] = []
        for lst in pool_aud.values():
            merged.extend(lst)
        pool_seg = merged
    if not pool_seg:
        return "Keluhan belajar sesuai jenjang saat ini, tanpa menyebut produk."
    rnd_seed = time.time_ns() ^ hash((audience, segment)) ^ random.getrandbits(32)
    random.seed(rnd_seed)
    return random.choice(pool_seg)

# === A5: Prompt untuk persona non-sales ======================================
def build_system_prompt(audience: str, segment: str) -> str:
    return "\n".join([
        f"Peran: Anda {audience} segmen {segment}.",
        "Tujuan: sampaikan masalah, konteks, harapan, dan batasan secara natural dari sudut pandang Anda.",
        "Gaya: 1–3 kalimat, natural, tanpa daftar bullet, tanpa jargon pemasaran, tanpa pengulangan.",
        "Interaksi: tutup dengan satu pertanyaan klarifikasi singkat bila relevan.",
        "Larangan keras: jangan menawarkan produk, jangan menyebut nama/kode paket, jangan menyarankan program, jangan ajak membeli, jangan pitching.",
        "Jika ditanya produk secara langsung: jawab tidak tahu detail produk; kembalikan fokus ke pengalaman pribadi dan kebutuhan.",
    ])

SEG_RULES = {
    "SD": "Hindari istilah Fisika, Kimia, UTBK; fokus literasi, numerasi dasar, IPA sederhana, kebiasaan belajar.",
    "SMP": "Hindari UTBK dan materi SMA; fokus aljabar dasar, IPA terapan, manajemen waktu.",
    "SMA": "Boleh UTBK dan materi lanjutan; hindari topik terlalu dasar SD/SMP.",
}

# === Filter frasa yang dilarang (CS-like, perkenalan diri, dan penjadwalan) ==
STOP_PHRASES = [
    "ada yang bisa saya bantu",
    "bagaimana saya bisa membantu",
    "bisa dibantu apa",
    "ada yang bisa dibantu",
    "apa yang bisa saya bantu",
    "bisa dihubungi",
    "bisa di hubungi",
    "bisa ditelepon",
    "bisa di telepon",
    "boleh telepon sekarang",
    "sekarang waktunya pas",
    "sekarang waktu yang pas",
    "waktunya pas",
    "ada waktu lain",
    "jadwal yang cocok",
    "kapan waktu yang tepat",
    "diskusi sebentar atau nanti",
    "apakah sekarang waktu",
    "ini bundanya",
    "ini ayahnya",
    "ini ibunya",
    "ini orang tua",
    "ini orangtua",
]

def build_dialog_instruction(audience: str, segment: str) -> str:
    rule = SEG_RULES.get(segment, "")
    banned = "; ".join(STOP_PHRASES)
    return (
        f"Anda tetap berperan sebagai {audience} segmen {segment}. "
        f"Patuh aturan segmen: {rule} "
        "Tanggapi ketat sesuai konteks pesan terakhir. Tidak menawarkan bantuan. Tidak promosi produk. "
        "Tidak menyebut ketersediaan, tidak menanyakan waktu/jadwal, tidak memperkenalkan identitas diri proaktif. "
        f"Hindari frasa: {banned}. "
        "Jika pesan pengguna hanya sapaan/cek identitas, balas singkat dan netral tanpa perkenalan diri, contoh: "
        "'Halo juga, ada apa ya?' atau 'Halo, siapa ya?'. "
        "Gunakan orang pertama konsisten sesuai persona hanya jika ditanya."
    )

def build_opener_instruction(audience: str, segment: str) -> str:
    scenario = st.session_state.get("opener_scenario") or _sample_scenario(audience, segment)
    rule = SEG_RULES.get(segment, "")
    return (
        f"Buat pembuka percakapan 1–2 kalimat sebagai {audience} segmen {segment}. "
        f"Gunakan skenario: {scenario}. "
        f"Patuh aturan segmen: {rule} "
        "Natural, tanpa menyebut produk atau paket. Variasikan diksi agar berbeda setiap kali."
    )

def recommend(segment: str, signals: Dict) -> List[Dict]:
    pilihan = CATALOG.get(segment, [])
    hasil = []
    for item in pilihan:
        skor = 0
        if signals.get("fokus_ujian") and "UTBK" in item["nama"]:
            skor += 3
        if signals.get("butuh_live") and ("Live" in item["nama"] or any("Live" in f for f in item["fitur"])):
            skor += 2
        if signals.get("butuh_latihan") and any(("Bank soal" in f) or ("Tryout" in f) for f in item["fitur"]):
            skor += 2
        if segment == "SD" and "Video" in item["nama"] and signals.get("konsep_dasar"):
            skor += 2
        if segment in ["SMP", "SMA"] and "Paket" in item["nama"]:
            skor += 1
        hasil.append({"skor": skor, **item})
    hasil.sort(key=lambda x: x["skor"], reverse=True)
    return hasil[:3]

# === A6: Sidebar/state ========================================================
with st.sidebar:
    st.title("RG Telesales - Role-Play")
    _aud_opts = ["Orang Tua", "Murid"]
    _seg_opts = ["SD", "SMP", "SMA"]
    _aud_idx = _aud_opts.index(st.session_state.get("aud", "Orang Tua")) \
        if st.session_state.get("aud") in _aud_opts else 0
    _seg_idx = _seg_opts.index(st.session_state.get("seg", "SMP")) \
        if st.session_state.get("seg") in _seg_opts else 1
    audience = st.selectbox("Peran lawan bicara", _aud_opts, index=_aud_idx, key="aud")
    segment  = st.selectbox("Segmen kelas", _seg_opts, index=_seg_idx, key="seg")
    if os.getenv("SHOW_MODEL_INFO") == "1":
        st.caption(f"SDK: {SDK} | Model: {MODEL_PRIMARY}")
    st.divider()
    st.subheader("Riwayat Chat", anchor=False)
    convos = list_convos()
    labels = [
        f"{c['title'][:40]} · {c['audience'] or '-'}-{c['segment'] or '-'} · {c['updated_at'][5:16]}"
        for c in convos
    ] or ["(belum ada)"]
    sel_idx = st.selectbox(
        "Semua sesi",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        index=0 if convos else 0,
        key="hist_select_idx",
    )
    st.caption("Aksi riwayat")
    cols = st.columns([1, 1, 1], gap="small")
    with cols[0]:
        if st.button("Buka", key="btn_open_hist", use_container_width=True) and convos:
            load_convo(convos[sel_idx]["id"])
    with cols[1]:
        if st.button("Sesi Baru", key="btn_new_session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.internal_triggers = []
            st.session_state.intent = None
            st.session_state.suppress_next_reply = True
            for k in ["convo_id", "convo_title"]:
                st.session_state.pop(k, None)
    with cols[2]:
        if st.button("Hapus", key="btn_delete_hist", use_container_width=True) and convos:
            delete_convo(convos[sel_idx]["id"])

# === State init + autosave awal ==============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "signals" not in st.session_state:
    st.session_state.signals = {"fokus_ujian": False, "butuh_live": False, "butuh_latihan": True, "konsep_dasar": False}
if "suppress_next_reply" not in st.session_state:
    st.session_state.suppress_next_reply = False
if "intent" not in st.session_state:
    st.session_state.intent = None  # None | "opener"
if "internal_triggers" not in st.session_state:
    st.session_state.internal_triggers = []  # penanda pesan sintetis
if "bot_persona" not in st.session_state:
    st.session_state.bot_persona = audience
if "opener_scenario" not in st.session_state:
    st.session_state.opener_scenario = None

# autosave jika sudah ada pesan tetapi belum memiliki convo_id
if st.session_state.get("messages") and not st.session_state.get("convo_id"):
    save_current_convo()

def get_effective_audience() -> str:
    return st.session_state.get("bot_persona", st.session_state.get("aud", "Orang Tua"))

# Auto tuning (tanpa kontrol UI)
def current_temperature() -> float:
    return 0.35 if st.session_state.get("intent") == "opener" else 0.3

def history_window() -> int:
    n = len(st.session_state.get("messages", []))
    if st.session_state.get("intent") == "opener":
        return 0  # abaikan riwayat saat pembuka agar variasi tidak terikat konteks lama
    if n <= 6:
        return 6
    if n <= 12:
        return 8
    return 12

# === A6b: Deteksi sapaan minimal dan jawaban ringkas ==========================
GREETING_PATTERNS = [
    r"^\s*(halo|hai|hi|helo|hello)\s*!?\s*$",
    r"^\s*(halo|hai|hi|helo|hello)\s+kak\b.*$",
    r"^\s*(ass?alam(u|)alaikum)(\s+wr\.?\s*wb\.?)?\s*$",
    r"^\s*(pagi|siang|sore|malam)\s*!?\s*$",
]
GREETING_REPLIES = [
    "Halo juga, ada apa ya?",
    "Halo, ada keperluan apa?",
    "Halo juga, bisa disampaikan maksudnya?",
    "Halo, ada yang ingin dibahas?",
]

def _is_minimal_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) <= 20:
        for pat in GREETING_PATTERNS:
            if re.match(pat, t):
                return True
    return False

# === A7: Opener ===============================================================
def _trigger_model_opener(target_audience: str):
    st.session_state.bot_persona = target_audience
    st.session_state.opener_scenario = _sample_scenario(target_audience, st.session_state.get("seg", "SMP"))
    st.session_state.intent = "opener"
    ping = "⏩ OPENER"
    st.session_state.messages.append({"role": "user", "content": ping})
    st.session_state.internal_triggers.append(ping)
    st.session_state.suppress_next_reply = False
    save_current_convo()  # autosave perubahan intent/triggers

c1, c2 = st.columns(2)
if c1.button("Opener Orang Tua"):
    _trigger_model_opener("Orang Tua")
if c2.button("Opener Murid"):
    _trigger_model_opener("Murid")

# === A8: Header + rekomendasi cepat (UI informatif) ==========================
st.markdown("## Chat Role-Play")
sys_prompt = build_system_prompt(get_effective_audience(), segment)
# Rekomendasi cepat dinonaktifkan

# === A9: Input sebelum render chat ===========================================
user_input = st.chat_input("Ketik pesan Anda di sini")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.suppress_next_reply = False
    save_current_convo()  # autosave setelah input

# === A10: Render riwayat (avatar dibedakan) ==================================
AVATAR_USER = "🧑"

def _bot_avatar(aud: str) -> str:
    return "🧑‍🦳" if aud == "Orang Tua" else "🧑‍🎓"

h = history_window()
for m in st.session_state.messages[-h or None:]:
    if m["content"] in st.session_state.internal_triggers:
        continue
    role = "user" if m["role"] == "user" else "assistant"
    avatar = AVATAR_USER if role == "user" else _bot_avatar(get_effective_audience())
    with st.chat_message(role, avatar=avatar):
        st.markdown(m["content"])

# === A11: Prompt composer =====================================================
def build_prompt(messages: List[Dict], audience: str, segment: str, opener: bool = False) -> str:
    meta = {
        "audience": audience,
        "segment": segment,
        "time": datetime.now().isoformat(timespec="seconds"),
        "policy": {"no_pitch": True, "no_product_names": True, "focus_persona": True},
        "mode": "opener" if opener else "dialog",
        "nonce": int(time.time() * 1000),
    }
    history_lines = []
    limit = history_window()
    if not opener and limit:
        for m in messages[-limit:]:
            if m["content"] in st.session_state.internal_triggers:
                continue
            role = "User" if m["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {m['content']}")
    convo = "\n".join(history_lines)
    task = build_opener_instruction(audience, segment) if opener else build_dialog_instruction(audience, segment)
    return (
        f"[META]\n{json.dumps(meta, ensure_ascii=False)}\n\n"
        f"[SYSTEM]\n{build_system_prompt(audience, segment)}\n\n"
        f"[HISTORY]\n{convo}\n\n"
        f"[TASK]\n{task}\n\n[RESPON]"
    )

# === A12: Safety settings (SDK baru) =========================================
def _safety_settings_new() -> List[Any]:
    return [
        types_new.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_MEDIUM_AND_ABOVE"),
        types_new.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_MEDIUM_AND_ABOVE"),
        types_new.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
        types_new.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    ]

# === A13: Safety settings (SDK lama) =========================================
def _safety_kwargs_legacy() -> dict:
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        pairs = [
            (HarmCategory.HARM_CATEGORY_HARASSMENT, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
            (HarmCategory.HARM_CATEGORY_HATE_SPEECH, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
            (HarmCategory.HARM_CATEGORY_SEXUAL_CONTENT, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
            (HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
        ]
        return {"safety_settings": [{"category": c, "threshold": t} for c, t in pairs]}
    except Exception:
        return {}

# === A14: Ekstraksi teks ======================================================
def _get_finish_reason(candidate: Any) -> Optional[str]:
    fr = getattr(candidate, "finish_reason", None) or getattr(candidate, "finishReason", None)
    return str(fr) if fr else None

def _extract_text_from_response(resp) -> str:
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    try:
        cands = getattr(resp, "candidates", None) or []
        pieces: List[str] = []
        for cand in cands:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for p in parts:
                tx = getattr(p, "text", None)
                if isinstance(tx, str) and tx:
                    pieces.append(tx)
        if pieces:
            return "".join(pieces).strip()
        if cands:
            fr = _get_finish_reason(cands[0])
            if fr and fr.upper() != "STOP":
                return f"Tidak ada keluaran karena finish_reason={fr}"
    except Exception:
        pass
    fb = getattr(resp, "prompt_feedback", None)
    if fb:
        reason = getattr(fb, "block_reason", None)
        if reason:
            return f"Tidak ada keluaran karena diblokir kebijakan: {reason}"
    return ""

def _extract_text_from_stream_event(event) -> Optional[str]:
    t = getattr(event, "text", None)
    if isinstance(t, str) and t:
        return t
    try:
        for cand in (getattr(event, "candidates", None) or []):
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for p in parts:
                tx = getattr(p, "text", None)
                if isinstance(tx, str) and tx:
                    return tx
    except Exception:
        pass
    return None

# === A15: Config builder (SDK baru) ==========================================
def _build_config_new(sys_text: str):
    return types_new.GenerateContentConfig(
        system_instruction=sys_text,
        temperature=float(current_temperature()),
        top_p=0.9,
        top_k=40,
        max_output_tokens=256,
        candidate_count=1,
        safety_settings=_safety_settings_new(),
    )

# === A16: Generator abstrak ===================================================
def generate_reply() -> str:
    opener_mode = st.session_state.intent == "opener"
    prompt = build_prompt(st.session_state.messages, get_effective_audience(), segment, opener=opener_mode)

    if SDK == "new":
        cfg = _build_config_new(sys_prompt)
        for model_name in MODEL_FALLBACKS:
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                area = st.empty()
                pieces: List[str] = []
                last_push = time.perf_counter()
                BATCH_CHARS = 160
                MIN_INTERVAL = 0.04
                pending = 0
                for chunk in stream:
                    piece = getattr(chunk, "text", None)
                    if not isinstance(piece, str) or not piece:
                        continue
                    pieces.append(piece)
                    pending += len(piece)
                    now = time.perf_counter()
                    if pending >= BATCH_CHARS or (now - last_push) >= MIN_INTERVAL:
                        area.markdown("".join(pieces))
                        pending = 0
                        last_push = now
                final = "".join(pieces).strip()
                if final:
                    area.markdown(final)
                    return final
            except Exception:
                continue
        last_reason = ""
        for model_name in MODEL_FALLBACKS:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                text = _extract_text_from_response(resp)
                if text:
                    return text
                cands = getattr(resp, "candidates", None) or []
                if cands:
                    fr = _get_finish_reason(cands[0])
                    if fr and fr.upper() != "STOP":
                        last_reason = f"finish_reason={fr}"
            except Exception as e:
                last_reason = f"{type(e).__name__}: {e}"
        return f"Model tidak mengembalikan teks. {last_reason or 'Coba ganti model/parameter.'}"

    safety_kw = _safety_kwargs_legacy()
    for model_name in MODEL_FALLBACKS:
        try:
            model = genai_legacy.GenerativeModel(model_name=model_name, **safety_kw)
            stream = model.generate_content(prompt, stream=True)
            area = st.empty()
            pieces: List[str] = []
            last_push = time.perf_counter()
            BATCH_CHARS = 160
            MIN_INTERVAL = 0.04
            pending = 0
            for event in stream:
                piece = _extract_text_from_stream_event(event)
                if not piece:
                    continue
                pieces.append(piece)
                pending += len(piece)
                now = time.perf_counter()
                if pending >= BATCH_CHARS or (now - last_push) >= MIN_INTERVAL:
                    area.markdown("".join(pieces))
                    pending = 0
                    last_push = now
            final = "".join(pieces).strip()
            if final:
                area.markdown(final)
                return final
        except Exception:
            continue
    last_reason = ""
    for model_name in MODEL_FALLBACKS:
        try:
            model = genai_legacy.GenerativeModel(model_name=model_name, **safety_kw)
            resp = model.generate_content(prompt, stream=False)
            text = _extract_text_from_response(resp)
            if text:
                return text
            cands = getattr(resp, "candidates", None) or []
            if cands:
                fr = _get_finish_reason(cands[0])
                if fr and fr.upper() != "STOP":
                    last_reason = f"finish_reason={fr}"
        except Exception as e:
            last_reason = f"{type(e).__name__}: {e}"
    return f"Model tidak mengembalikan teks. {last_reason or 'Coba ganti model/parameter.'}"

# === A17: Eksekusi balasan ====================================================
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and not st.session_state.suppress_next_reply
):
    last_user_msg = st.session_state.messages[-1]["content"]
    if st.session_state.intent != "opener" and _is_minimal_greeting(last_user_msg):
        reply = random.choice(GREETING_REPLIES)
        with st.chat_message("assistant", avatar=_bot_avatar(get_effective_audience())):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        save_current_convo()  # autosave setelah balasan singkat
    else:
        with st.chat_message("assistant", avatar=_bot_avatar(get_effective_audience())):
            reply = generate_reply()
            st.session_state.messages.append({"role": "assistant", "content": reply})
        save_current_convo()  # autosave setelah balasan model
    if st.session_state.intent == "opener":
        st.session_state.intent = None
        st.session_state.opener_scenario = None  # reset agar klik berikutnya sampling ulang

# === A18: Export transcript ====================================================
def to_markdown_transcript(msgs: List[Dict]) -> str:
    lines = ["# Transcript - RG Telesales Role-Play", ""]
    for m in msgs:
        if m["content"] in st.session_state.internal_triggers:
            continue
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}")
    return "\n\n".join(lines)

cA, cB = st.columns([1, 1])
with cA:
    if st.button("Unduh Transcript .md"):
        md = to_markdown_transcript(st.session_state.messages)
        st.download_button(
            "Download",
            data=md,
            file_name="transcript_rg_telesales.md",
            mime="text/markdown",
            use_container_width=True,
        )
with cB:
    st.caption("Privasi: hindari data sensitif saat role-play.")
