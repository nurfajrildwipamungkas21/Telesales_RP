# app.py
# =============================================================================
# Streamlit Chat - Telesales Ruangguru (Role-Play) + Gemini 2.5 Flash
# Migrasi penuh ke Google GenAI SDK:
# - Client terpusat: genai.Client()
# - Streaming: client.models.generate_content_stream(...)
# - Non-stream fallback: client.models.generate_content(...)
# - Config via types.GenerateContentConfig (system_instruction, safety_settings, sampling)
# - Nama model tanpa "models/" prefix
# =============================================================================
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

import streamlit as st

# Google GenAI SDK (sesuai migrasi)
# Ref: https://ai.google.dev/gemini-api/docs/migrate
from google import genai
from google.genai import types

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
</style>
""",
    unsafe_allow_html=True,
)

# === A1: API key dan client ===================================================
# SDK baru otomatis membaca GEMINI_API_KEY atau GOOGLE_API_KEY dari environment.
# Boleh juga diset eksplisit ke Client bila diperlukan.
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

@st.cache_resource(show_spinner=False)
def _init_client() -> genai.Client:
    return genai.Client(api_key=API_KEY) if API_KEY else genai.Client()

client = _init_client()

# Model utama + fallback sesuai ketersediaan akun/region
MODEL_PRIMARY = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_FALLBACKS = [
    MODEL_PRIMARY,
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# === A2: Katalog dummy ========================================================
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

# === A3: Prompt builder =======================================================
def build_system_prompt(audience: str, segment: str) -> str:
    return "\n".join([
        "Etik: sopan, informatif, tanpa janji palsu, tidak memaksa, tidak meminta data sensitif.",
        "Tujuan: deteksi kebutuhan belajar, jelaskan fitur relevan, tawarkan sesi lanjutan dengan persetujuan.",
        "Gaya: ringkas, langkah demi langkah, pertanyaan tertutup lalu terbuka, hindari jargon.",
        f"Peran Anda: Chatbot yang memerankan calon {audience} segmen {segment}.",
        "Larangan: jangan janji nilai, jangan minta data pembayaran, hindari SARA."
    ])

# === A4: Rekomendasi sederhana ===============================================
def recommend(segment: str, signals: Dict) -> List[Dict]:
    pilihan = CATALOG.get(segment, [])
    hasil = []
    for item in pilihan:
        skor = 0
        if signals.get("fokus_ujian") and "UTBK" in item["nama"]: skor += 3
        if signals.get("butuh_live") and ("Live" in item["nama"] or any("Live" in f for f in item["fitur"])): skor += 2
        if signals.get("butuh_latihan") and any(("Bank soal" in f) or ("Tryout" in f) for f in item["fitur"]): skor += 2
        if segment == "SD" and "Video" in item["nama"] and signals.get("konsep_dasar"): skor += 2
        if segment in ["SMP", "SMA"] and "Paket" in item["nama"]: skor += 1
        hasil.append({"skor": skor, **item})
    hasil.sort(key=lambda x: x["skor"], reverse=True)
    return hasil[:3]

# === A5: Sidebar ==============================================================
with st.sidebar:
    st.title("RG Telesales - Role-Play")
    audience = st.selectbox("Peran lawan bicara", ["Orang Tua", "Murid"], index=0, key="aud")
    segment = st.selectbox("Segmen kelas", ["SD", "SMP", "SMA"], index=1, key="seg")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1, key="temp")
    max_history = st.slider("Batas riwayat pesan", 4, 32, 8, 2, key="hist")
    st.caption(f"Model: {MODEL_PRIMARY}")
    st.caption("API key dibaca dari GEMINI_API_KEY/GOOGLE_API_KEY (env)")

# === A6: Session state ========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "signals" not in st.session_state:
    st.session_state.signals = {"fokus_ujian": False, "butuh_live": False, "butuh_latihan": True, "konsep_dasar": False}
if "suppress_next_reply" not in st.session_state:
    st.session_state.suppress_next_reply = False

# === A7: Preset openers - tidak auto-reply ===================================
c1, c2, c3 = st.columns(3)
if c1.button("Opener Orang Tua"):
    st.session_state.messages.append({
        "role": "user",
        "content": "Halo, saya orang tua. Anak saya sering bingung saat latihan matematika dan nilainya turun."
    })
    st.session_state.suppress_next_reply = True
if c2.button("Opener Murid"):
    st.session_state.messages.append({
        "role": "user",
        "content": "Kak, aku kelas 9. Aku pengin naik nilai IPA, tapi suka keteteran di fisika."
    })
    st.session_state.suppress_next_reply = True
if c3.button("Minta Rekomendasi"):
    st.session_state.messages.append({
        "role": "user",
        "content": "Bisa kasih rekomendasi paket belajar yang cocok dengan kebutuhan saya?"
    })
    st.session_state.suppress_next_reply = True

# === A8: Header dan rekomendasi ==============================================
st.markdown("## Chat Role-Play")
sys_prompt = build_system_prompt(audience, segment)
top_reco = recommend(segment, st.session_state.signals)
with st.expander("Rekomendasi cepat - dinamis, non-binding"):
    for r in top_reco:
        st.write(f"{r['nama']} - fitur: {', '.join(r['fitur'])} | cocok: {', '.join(r['cocok'])}")

# === A9: Input pengguna sebelum render chat ==================================
user_input = st.chat_input("Ketik pesan Anda di sini")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.suppress_next_reply = False

# === A10: Render riwayat dengan avatar valid =================================
AVATAR_USER = "👤"
AVATAR_BOT  = "🤖"

for m in st.session_state.messages[-max_history:]:
    role = "user" if m["role"] == "user" else "assistant"
    avatar = AVATAR_USER if role == "user" else AVATAR_BOT
    with st.chat_message(role, avatar=avatar):
        st.markdown(m["content"])

# === A11: Prompt composer =====================================================
def build_prompt(messages: List[Dict], audience: str, segment: str) -> str:
    meta = {
        "audience": audience,
        "segment": segment,
        "time": datetime.now().isoformat(timespec="seconds"),
        "policy": {"no_fake_promises": True, "no_sensitive_data": True, "no_payment_request": True},
    }
    history_lines = []
    for m in messages[-max_history:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history_lines.append(f"{role}: {m['content']}")
    convo = "\n".join(history_lines)
    return (
        f"[META]\n{json.dumps(meta, ensure_ascii=False)}\n\n"
        f"[CATALOG]\n{json.dumps(CATALOG.get(segment, []), ensure_ascii=False)}\n\n"
        f"[CONTEXT]\nPercakapan sebelumnya:\n{convo}\n\n[RESPON]"
    )

# === A12: Safety settings (SDK baru) =========================================
def _safety_settings() -> List[types.SafetySetting]:
    # Ref: safety_settings via config di SDK baru
    return [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_MEDIUM_AND_ABOVE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_MEDIUM_AND_ABOVE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    ]

# === A13: Util ekstraksi teks dan alasan =====================================
def _get_finish_reason(candidate: Any) -> Optional[str]:
    # Robust terhadap snake_case / camelCase
    fr = getattr(candidate, "finish_reason", None)
    if not fr:
        fr = getattr(candidate, "finishReason", None)
    return str(fr) if fr else None

def _extract_text_from_response(resp) -> str:
    # Utama: resp.text
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # Alternatif: gabung parts
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
        # Tidak ada teks: kembalikan alasan bila ada
        if cands:
            fr = _get_finish_reason(cands[0])
            if fr and fr.upper() != "STOP":
                return f"Tidak ada keluaran karena finish_reason={fr}"
    except Exception:
        pass
    return ""

# === A14: Generate - streaming via GenAI SDK + fallback ======================
def _build_config(sys_text: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=sys_text,             # system instruction via config (SDK baru)
        temperature=float(temperature),
        top_p=0.9,
        top_k=40,
        max_output_tokens=256,
        candidate_count=1,
        safety_settings=_safety_settings(),
    )

def generate_reply() -> str:
    prompt = build_prompt(st.session_state.messages, audience, segment)
    cfg = _build_config(sys_prompt)

    # 1) Streaming attempt berurutan dengan fallback model
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
            # lanjut ke model fallback berikut
            continue

    # 2) Non-stream fallback: ambil first non-empty text atau alasan blokir
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
            # simpan alasan bila ada untuk pelaporan akhir
            cands = getattr(resp, "candidates", None) or []
            if cands:
                fr = _get_finish_reason(cands[0])
                if fr and fr.upper() != "STOP":
                    last_reason = f"finish_reason={fr}"
        except Exception as e:
            last_reason = f"{type(e).__name__}: {e}"

    return f"Model tidak mengembalikan teks. {last_reason or 'Coba ganti model/parameter.'}"

# === A15: Eksekusi balasan jika bukan opener =================================
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and not st.session_state.suppress_next_reply
):
    with st.chat_message("assistant", avatar=AVATAR_BOT):
        reply = generate_reply()
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.markdown(reply)

# === A16: Export transcript ===================================================
def to_markdown_transcript(msgs: List[Dict]) -> str:
    lines = ["# Transcript - RG Telesales Role-Play", ""]
    for m in msgs:
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

st.caption("Portfolio demo. Streaming dibatch, fallback aktif, opener tanpa auto-reply, avatar dibedakan. SDK: google-genai.")
