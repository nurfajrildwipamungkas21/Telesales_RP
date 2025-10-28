# app.py
# =============================================================================
# Streamlit Chat — Telesales Ruangguru (Role-Play) + Gemini 2.5 Flash
# Optimized: cached model, batched streaming, opener tidak auto-reply,
# urutan render diperbaiki agar chat user langsung tampil, perbaikan layout
# =============================================================================
import json
import time
from datetime import datetime
from typing import List, Dict

import streamlit as st
import google.generativeai as genai

# === A0: Page config + CSS ====================================================
st.set_page_config(page_title="RG Telesales — Role-Play Chat", layout="wide")
st.markdown(
    """
<style>
/* Ruang atas agar header tidak terpotong di berbagai perangkat */
.block-container {padding-top: 2.25rem; padding-bottom: 1rem; max-width: 980px;}
/* Chat bubble rapat dan stabil */
.stChatMessage {gap: .25rem;}
/* Scroll halus untuk mobile/iOS */
html, body {scroll-behavior: smooth;}
@media (max-width: 600px){
  .block-container {padding-left: .6rem; padding-right: .6rem;}
}
</style>
""",
    unsafe_allow_html=True,
)

# === A1: Kunci & Model (cached) ==============================================
API_KEY = "AIzaSyDd19AHP6cciyErg-bky3u07fXVGnXaraE"
MODEL_NAME = "models/gemini-2.5-flash"

@st.cache_resource(show_spinner=False)
def _init_model():
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel(model_name=MODEL_NAME)

model = _init_model()

# === A2: Katalog (dummy) ======================================================
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
        {"kode": "RB-SMA-LIVE", "nama": "RuangBelajar Live SMA", "fitur": ["Live STEM & Soshum", "Jadwal intensif"], "cocok": ["kelas 10-12", "persiapan ujian sekolah"]},
        {"kode": "RB-UTBK", "nama": "Paket UTBK", "fitur": ["Tryout UTBK", "Pembahasan SKD", "Strategi waktu"], "cocok": ["kelas 12", "fokus kampus tujuan"]}
    ]
}

# === A3: Prompt builder =======================================================
def build_system_prompt(audience: str, segment: str) -> str:
    return "\n".join([
        "Etik: sopan, informatif, tanpa janji palsu, tidak memaksa, tidak meminta data sensitif.",
        "Tujuan: deteksi kebutuhan belajar, jelaskan fitur relevan, tawarkan sesi lanjutan dengan persetujuan.",
        "Gaya: ringkas, langkah-demi-langkah, pertanyaan tertutup lalu terbuka, hindari jargon.",
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
    st.title("RG Telesales — Role-Play")
    audience = st.selectbox("Peran lawan bicara", ["Orang Tua", "Murid"], index=0, key="aud")
    segment = st.selectbox("Segmen kelas", ["SD", "SMP", "SMA"], index=1, key="seg")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1, key="temp")
    max_history = st.slider("Batas riwayat pesan", 4, 32, 8, 2, key="hist")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption("Kunci demo tertanam untuk portfolio")

# === A6: Session state ========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "signals" not in st.session_state:
    st.session_state.signals = {"fokus_ujian": False, "butuh_live": False, "butuh_latihan": True, "konsep_dasar": False}
if "suppress_next_reply" not in st.session_state:
    st.session_state.suppress_next_reply = False

# === A7: Preset openers (tidak auto-reply) ===================================
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

# === A8: Header + rekomendasi kontekstual ====================================
st.markdown("## Chat Role-Play")
sys_prompt = build_system_prompt(audience, segment)
top_reco = recommend(segment, st.session_state.signals)
with st.expander("Rekomendasi cepat (dinamis, non-binding)"):
    for r in top_reco:
        st.write(f"{r['nama']} — fitur: {', '.join(r['fitur'])} | cocok: {', '.join(r['cocok'])}")

# === A9: Input pengguna (harus sebelum render chat agar langsung muncul) =====
user_input = st.chat_input("Ketik pesan Anda di sini")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.suppress_next_reply = False

# === A10: Render riwayat ======================================================
for m in st.session_state.messages[-max_history:]:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.markdown(m["content"])

# === A11: Prompt composer (gabung string untuk stabilitas API) ===============
def build_prompt(messages: List[Dict], sys_text: str, audience: str, segment: str) -> str:
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
    prompt = (
        f"[META]\n{json.dumps(meta, ensure_ascii=False)}\n\n"
        f"[SYSTEM]\n{sys_text}\n"
        "Taktik: ringkas, mulai dengan 1-2 pertanyaan klarifikasi, akhiri opsi tindak lanjut tanpa menekan. "
        "Batas 6 baris.\n\n"
        f"[CATALOG]\n{json.dumps(CATALOG.get(segment, []), ensure_ascii=False)}\n\n"
        f"[CONTEXT]\nPercakapan sebelumnya:\n{convo}\n\n[RESPON]"
    )
    return prompt

# === A12: Generate — batched streaming, error robust =========================
def generate_reply():
    prompt = build_prompt(st.session_state.messages, sys_prompt, audience, segment)
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=float(temperature),
                top_p=0.9,
                top_k=40,
                max_output_tokens=256,
                candidate_count=1,
            ),
            stream=True,
        )
        area = st.empty()
        buffer = []
        last_push = time.perf_counter()
        BATCH_CHARS = 120
        MIN_INTERVAL = 0.03
        pending = 0

        for event in resp:
            chunk = getattr(event, "text", None)
            if not chunk:
                continue
            buffer.append(chunk)
            pending += len(chunk)
            now = time.perf_counter()
            if pending >= BATCH_CHARS or (now - last_push) >= MIN_INTERVAL:
                area.markdown("".join(buffer))
                pending = 0
                last_push = now

        final_text = "".join(buffer).strip()
        if final_text:
            area.markdown(final_text)
        return final_text or " "
    except Exception as e:
        st.warning(f"Gagal memproses: {e}")
        return f"Gagal memproses: {e}"

# === A13: Eksekusi balasan (hanya jika bukan opener) =========================
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and not st.session_state.suppress_next_reply
):
    with st.chat_message("assistant"):
        reply = generate_reply()
        st.session_state.messages.append({"role": "assistant", "content": reply})

# === A14: Export transcript ===================================================
def to_markdownTranscript(msgs: List[Dict]) -> str:
    lines = ["# Transcript — RG Telesales Role-Play", ""]
    for m in msgs:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"**{who}:** {m['content']}")
    return "\n\n".join(lines)

cA, cB = st.columns([1, 1])
with cA:
    if st.button("Unduh Transcript .md"):
        md = to_markdownTranscript(st.session_state.messages)
        st.download_button(
            "Download",
            data=md,
            file_name="transcript_rg_telesales.md",
            mime="text/markdown",
            use_container_width=True,
        )
with cB:
    st.caption("Privasi: hindari data sensitif saat role-play.")

st.caption("Portfolio demo. Optimized streaming, UI ringan, opener tanpa auto-reply.")
