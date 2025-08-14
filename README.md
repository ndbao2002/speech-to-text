
---

# 🎤 Speech-to-Text Screening Project

This repository contains a complete **speech-to-text pipeline** using **Mozilla DeepSpeech** and the **LibriSpeech Dev Set (SLR12)**.  
It was developed as part of a screening task and includes:

- **Dataset Evaluation** — Measure model accuracy on LibriSpeech
- **Noise Robustness Testing** — Add environmental noise & test WER
- **Interactive Web App** — Built with Streamlit for easy use
- **Noise Reduction** — Preprocess audio before STT
- **LLM Post-Processing** — Use Cohere API to improve transcriptions

---

## 🚀 Features

- 🎙 **Record** audio directly from browser
- 🔊 **Playback** original & denoised audio
- 📜 **Transcribe** using Mozilla DeepSpeech
- 🌐 **Improve** text via Cohere LLM
- 🧪 **Test** robustness with environmental noise

---

## 📦 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/stt-screening.git
cd stt-screening
```

### 2️⃣ Create Python Environment

Using **conda**:

```bash
conda create -n stt-screening python=3.9
conda activate stt-screening
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Download Resources 
Run download script:

```bash
python download.py
```

Or download manually:

### 1️⃣ DeepSpeech Pretrained Model

```bash
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
```

### 2️⃣ LibriSpeech Dataset (Dev Set)

```bash
wget https://openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
```



---

## 🧪 1. Offline Evaluation

Evaluate DeepSpeech accuracy on LibriSpeech (clean):

```bash
python eval.py
```

I also added a script for comparing acoustic models with and without scorer:
```bash
python comparison.py
```
This will output the comparison results between the models. Showing that with scorer, the sentence more matching the reference transcript.

---

## 🌐 2. Streamlit Web App

Run the live transcription app:

```bash
streamlit run app/app_streamlit_stt.py
```

Features:

* Record from microphone
* Playback original & denoised versions
* Transcribe with DeepSpeech
* Improve text with Cohere

---

## 🤖 3. LLM Post-Processing with Cohere

### 1️⃣ Add API Key

Create `app/.streamlit/secrets.toml`:

```toml
COHERE_API_KEY = "your_api_key_here"
```

### 2️⃣ Run the app again, now it will have LLM part

```bash
streamlit run app/app_streamlit_stt.py
```

---

## 🛠 Requirements

* Python 3.8–3.10
* Mozilla DeepSpeech 0.9.3
* Streamlit >= 1.20
* streamlit-webrtc
* NumPy, SciPy, SoundFile
* Cohere Python SDK

Install all with:

```bash
pip install -r requirements.txt
```

---

## 💬 Discussion

The original screening task required:
- Choosing a dataset (I used LibriSpeech Dev Set SLR12)
- Using an STT engine (I chose Mozilla DeepSpeech)
- Documenting results

I went **beyond the bare minimum** by implementing additional features and experiments:

### 1. Real-Time Web Application
Instead of just offline transcription, I built a **Streamlit** web app with:
- Live microphone recording from the browser
- Playback of both **original** and **denoised** audio
- Integration with DeepSpeech for real-time transcription
- Optional LLM post-processing with Cohere to improve grammar and punctuation

### 2. Noise Reduction Pipeline
I implemented an optional denoising step to improve transcription in noisy environments. I used `noisereduce`, so the technique is spectral gating.
This directly addresses the screening’s suggestion on “how to alleviate noise”.


### 3. Language Model Post-Processing
I incorporated **Cohere’s LLM** for gathering insights from the transcriptions.
This enhances readability without changing the meaning of the transcript.

### 4. Extensibility
Our modular design allows easy replacement of:
- **STT backend** (DeepSpeech → Whisper, etc.)
- **Noise reduction algorithm**
- **Post-processing LLM**

### 5. Resource-Constrained Deployment Considerations
Although not fully implemented, I discussed optimizations for low-power devices:
- Model quantization (reducing size and inference cost)
- Streaming chunk-based inference to avoid large memory usage
- Using smaller acoustic models with an external language model for improved WER

---

