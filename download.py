import os
import tarfile
import urllib.request

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DEEPSPEECH_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"
DEEPSPEECH_SCORER_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "deepspeech-0.9.3-models.pbmm")
SCORER_PATH = os.path.join(DATA_DIR, "deepspeech-0.9.3-models.scorer")
LIBRISPEECH_DIR = os.path.join(DATA_DIR, "LibriSpeech/dev-clean")



def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Already exists: {dest}")

os.makedirs(DATA_DIR, exist_ok=True)

# Download LibriSpeech dev-clean
libri_tgz = os.path.join(DATA_DIR, "dev-clean.tar.gz")
download_file(LIBRISPEECH_URL, libri_tgz)

# Extract
if not os.path.exists(LIBRISPEECH_DIR):
    print("Extracting LibriSpeech...")
    with tarfile.open(libri_tgz, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
else:
    print("LibriSpeech already extracted.")

# Download DeepSpeech model & scorer
download_file(DEEPSPEECH_MODEL_URL, MODEL_PATH)
download_file(DEEPSPEECH_SCORER_URL, SCORER_PATH)