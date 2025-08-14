import os
import numpy as np
import soundfile as sf
from deepspeech import Model

# --------------------------
# Paths
# --------------------------
MODEL_PATH = "data/deepspeech-0.9.3-models.pbmm"
SCORER_PATH = "data/deepspeech-0.9.3-models.scorer"

# Example FLAC from LibriSpeech
AUDIO_FILE = "data/LibriSpeech/dev-clean/1462/170138/1462-170138-0000.flac"
GROUND_TRUTH_FILE = "data/LibriSpeech/dev-clean/1462/170138/1462-170138.trans.txt"

# Read ground truth transcription
def get_ground_truth():
    target_id = os.path.basename(AUDIO_FILE).replace(".flac", "")

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()

        for line in ground_truth.splitlines():
            if line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utt_id, transcript = parts
                    if utt_id == target_id:
                        return target_id, transcript.lower()
    return None, None


# Read FLAC and ensure 16kHz mono PCM
def load_audio(file_path):
    audio, sample_rate = sf.read(file_path, dtype='int16')
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)  # Convert to mono
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        audio = (audio * 32767).astype(np.int16)
        sample_rate = 16000
    return audio, sample_rate

# Run DeepSpeech inference
def transcribe_audio(audio, sample_rate, model_path, scorer_path=None):
    ds = Model(model_path)
    if scorer_path:
        ds.enableExternalScorer(scorer_path)
    return ds.stt(audio)

# Compare results
audio, sr = load_audio(AUDIO_FILE)

ground_truth_id, ground_truth = get_ground_truth()
result_no_scorer = transcribe_audio(audio, sr, MODEL_PATH)
result_with_scorer = transcribe_audio(audio, sr, MODEL_PATH, SCORER_PATH)

print("\n=== COMPARISON ===")
print("Ground Truth   :", ground_truth)
print("Without scorer:", result_no_scorer)
print("With scorer   :", result_with_scorer)
