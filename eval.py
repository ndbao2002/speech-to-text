import os
import numpy as np
import soundfile as sf
from deepspeech import Model
from jiwer import wer
import librosa
import tqdm

# Paths
MODEL_PATH = "data/deepspeech-0.9.3-models.pbmm"
SCORER_PATH = "data/deepspeech-0.9.3-models.scorer"
LIBRISPEECH_DIR = "data/LibriSpeech/dev-clean"

# Audio Loader (FLAC → PCM int16, mono, 16kHz)
def load_audio(file_path):
    audio, sample_rate = sf.read(file_path, dtype='int16')
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)  # stereo → mono
    if sample_rate != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        audio = (audio * 32767).astype(np.int16)
        sample_rate = 16000
    return audio, sample_rate

# DeepSpeech Transcription
def transcribe_audio(audio, model, use_scorer=False):
    if use_scorer:
        model.enableExternalScorer(SCORER_PATH)
    else:
        model.disableExternalScorer()
    return model.stt(audio)

# Main Evaluation
def evaluate_librispeech():
    # Load DeepSpeech model once
    ds_model = Model(MODEL_PATH)

    refs_no_scorer, hyps_no_scorer = [], []
    refs_with_scorer, hyps_with_scorer = [], []

    print("Starting evaluation over LibriSpeech dev-clean...\n")

    for root, _, files in os.walk(LIBRISPEECH_DIR):
        # Find the .trans.txt file in this directory (if any)
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        trans_dict = {}
        if trans_files:
            trans_path = os.path.join(root, trans_files[0])
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        trans_dict[parts[0]] = parts[1]

        for fname in tqdm.tqdm(files):
            if fname.endswith(".flac"):
                audio_path = os.path.join(root, fname)
                utt_id = fname.replace(".flac", "")
                if utt_id not in trans_dict:
                    continue
                ground_truth = trans_dict[utt_id]

                # Load audio
                audio, sr = load_audio(audio_path)

                # No scorer transcription
                pred_no_scorer = transcribe_audio(audio, ds_model, use_scorer=False)
                refs_no_scorer.append(ground_truth.lower())
                hyps_no_scorer.append(pred_no_scorer.lower())

                # With scorer transcription
                pred_with_scorer = transcribe_audio(audio, ds_model, use_scorer=True)
                refs_with_scorer.append(ground_truth.lower())
                hyps_with_scorer.append(pred_with_scorer.lower())

    # Calculate WER
    wer_no_scorer = wer(refs_no_scorer, hyps_no_scorer)
    wer_with_scorer = wer(refs_with_scorer, hyps_with_scorer)

    print("\n=== Final Evaluation Results ===")
    print(f"WER without scorer: {wer_no_scorer:.4f}")
    print(f"WER with scorer   : {wer_with_scorer:.4f}")

    # Save results to file
    with open("eval_results.txt", "w") as f:
        f.write(f"WER without scorer: {wer_no_scorer:.4f}\n")
        f.write(f"WER with scorer   : {wer_with_scorer:.4f}\n")

if __name__ == "__main__":
    evaluate_librispeech()
