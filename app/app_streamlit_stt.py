import os
import io
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
from deepspeech import Model
from llm_cohere import cohere_postprocess

from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# ---------- Settings ----------
MODEL_PATH = "data/deepspeech-0.9.3-models.pbmm"
SCORER_PATH = "data/deepspeech-0.9.3-models.scorer"

st.set_page_config(page_title="Speech → Text Demo", page_icon="🎙️", layout="centered")
st.title("🎙️ Speech-to-Text Demo (Upload or Record)")

with st.sidebar:
    st.header("Settings")
    use_scorer = st.checkbox("Use language model (.scorer)", value=True)
    do_denoise = st.checkbox("Apply noise reduction", value=True)
    do_llm = st.checkbox("Post-process with Cohere LLM", value=True)
    st.caption("Cohere key should be in .streamlit/secrets.toml")

# ---------- Utilities ----------
@st.cache_resource(show_spinner=True)
def load_model(model_path, scorer_path, enable_scorer=True):
    m = Model(model_path)
    if enable_scorer and os.path.exists(scorer_path):
        m.enableExternalScorer(scorer_path)
    return m

def ensure_16k_mono_float32(data, sr):
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    data = np.clip(data.astype(np.float32), -1.0, 1.0)
    return data, sr

def float32_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

def maybe_denoise(x, sr, enabled):
    if not enabled:
        return x
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=x, sr=sr).astype(np.float32)
    except Exception as e:
        st.warning(f"Noise reduction unavailable ({e})")
        return x

def transcribe(model, audio_float32, enable_scorer):
    try:
        if enable_scorer and os.path.exists(SCORER_PATH):
            model.enableExternalScorer(SCORER_PATH)
        else:
            model.disableExternalScorer()
    except Exception:
        pass
    audio_i16 = float32_to_int16(audio_float32)
    return model.stt(audio_i16)

def process_and_show(text):
    st.subheader("Transcript")
    st.write(text)
    if do_llm and text.strip():
        with st.spinner("Generating insights with Cohere…"):
            result = cohere_postprocess(text)
        if result:
            st.subheader("Cohere Insights")
            st.write(result)

# ---------- Upload flow ----------
st.markdown("## 📁 Upload an audio file")
uploaded = st.file_uploader("Choose .wav / .flac / .mp3", type=["wav", "flac", "mp3"])
if uploaded:
    # Try soundfile first (fast path), fallback to librosa for exotic mp3s
    try:
        data, sr = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float32")
    except Exception:
        y, sr = librosa.load(io.BytesIO(uploaded.getvalue()), sr=None, mono=False)
        data = y if y.ndim == 1 else y.mean(axis=0)

    st.audio(uploaded, format="audio/wav")
    data, sr = ensure_16k_mono_float32(data, sr)
    if do_denoise:
        data = maybe_denoise(data, sr, True)

    with st.spinner("Loading model…"):
        model = load_model(MODEL_PATH, SCORER_PATH, enable_scorer=use_scorer)
    with st.spinner("Transcribing…"):
        text = transcribe(model, data, enable_scorer=use_scorer)
    process_and_show(text)

st.markdown("---")

# ---------- Microphone recording (new API, stop + playback) ----------
st.markdown("## 🎤 Record from your microphone")

from av import AudioFrame
from typing import List
import io
import soundfile as sf

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer: List[np.ndarray] = []
        self.last_pts = None  # Last presentation timestamp

    def _append_audio(self, frame: AudioFrame):
        # Skip duplicate frames based on timestamp
        if self.last_pts is not None and frame.pts == self.last_pts:
            return
        self.last_pts = frame.pts

        arr = frame.to_ndarray()  # default layout works everywhere
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # mono
        arr = arr.astype(np.float32) / 32768.0
        self.buffer.append(arr.copy())

    def recv(self, frame: AudioFrame) -> AudioFrame:
        self._append_audio(frame)
        return frame

    async def recv_queued(self, frames: List[AudioFrame]) -> List[AudioFrame]:
        for frame in frames:
            self._append_audio(frame)
        return frames

def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono audio to WAV bytes for playback."""
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format='WAV')
    buf.seek(0)
    return buf.read()

ctx = webrtc_streamer(
    key="mic-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=MicAudioProcessor,
)

col1, col2 = st.columns(2)
clicked = col1.button("Stop & Transcribe", disabled=(ctx.audio_processor is None))
clear_clicked = col2.button("Clear buffer", disabled=(ctx.audio_processor is None))

if ctx.audio_processor is not None:
    if clear_clicked:
        ctx.audio_processor.buffer = []
        st.info("Audio buffer cleared.")

    if clicked:
        try:
            ctx.stop()
        except Exception:
            pass

        chunks = ctx.audio_processor.buffer
        if not chunks:
            st.error("❌ No audio detected — check mic permissions and try again.")
        else:
            audio = np.concatenate(chunks, axis=0)

            recorded_sr = 48000  # browser default

            # Resample to 16kHz mono
            audio_16k, _ = ensure_16k_mono_float32(audio, recorded_sr)

            # Playback original
            st.subheader("🔊 Original Audio")
            st.audio(audio_to_wav_bytes(audio, recorded_sr), format="audio/wav")

            # Denoise
            if do_denoise:
                audio_denoised = maybe_denoise(audio, recorded_sr, True)
                st.subheader("🔊 Denoised Audio")
                st.audio(audio_to_wav_bytes(audio_denoised, recorded_sr), format="audio/wav")
            else:
                audio_denoised = audio_16k

            # Transcribe
            with st.spinner("Loading model…"):
                model = load_model(MODEL_PATH, SCORER_PATH, enable_scorer=use_scorer)
            with st.spinner("Transcribing…"):
                text = transcribe(model, audio_denoised, enable_scorer=use_scorer)

            process_and_show(text)

            ctx.audio_processor.buffer = []