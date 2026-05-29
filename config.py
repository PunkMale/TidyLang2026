# config.py
import os

# ==========================================
# 1. Encoder and classification head
# ==========================================
# Encoder: ECAPA-TDNN (pre-trained on VoxLingua107) or XLS-R
MODEL_NAME_OR_PATH = "speechbrain/lang-id-voxlingua107-ecapa"
# MODEL_NAME_OR_PATH = "facebook/wav2vec2-xls-r-300m"

# Classification loss: "aam" (AAM-Softmax) or "ram" (RAM-Softmax)
LOSS_TYPE = "ram"

TARGET_SAMPLE_RATE = 16000
SAFE_MODEL_NAME = MODEL_NAME_OR_PATH.replace("/", "_")

# ==========================================
# 2. Training hyper-parameters
# ==========================================
EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_WORKERS = 16

# ==========================================
# 3. Data augmentation
# ==========================================
USE_AUGMENTATION = True
AUG_PROB = 0.8
# Root dirs of the noise / reverberation corpora. Edit to match your machine.
MUSAN_ROOT = "/data/noise/musan"
RIR_ROOT = "/data/noise/RIRS_NOISES/simulated_rirs"

AUG_SNR_MIN = 0.0
AUG_SNR_MAX = 15.0

# ==========================================
# 4. Audio root dirs (edit to match your machine)
# ==========================================
TRAIN_AUDIO_ROOT = "./data/TidyVoice/TidyVoiceX_Train"
VAL_AUDIO_ROOT = "./data/TidyVoice/TidyVoiceX_Dev"

MANIFEST_PATH = "./data/manifests/training_manifest.txt"

# Output dir is composed of encoder name + loss type + margin + augmentation flag
if USE_AUGMENTATION:
    SAVE_MODEL_DIR = f"./exp/{SAFE_MODEL_NAME}_{LOSS_TYPE}_0.2_augment"
else:
    SAVE_MODEL_DIR = f"./exp/{SAFE_MODEL_NAME}_{LOSS_TYPE}"

# Path of the dynamically built label dictionary
LABEL_DICT_PATH = os.path.join(SAVE_MODEL_DIR, "label_dict.pt")

SAVE_RESULT_DIR = f"{SAVE_MODEL_DIR}/results"
os.makedirs(SAVE_RESULT_DIR, exist_ok=True)

# ==========================================
# 5. Evaluation task paths
# ==========================================
# Task 1: language identification
TASK1_LID_TXT = "./eval_data/tl26_lid.txt"
TASK1_DATA_PATH = "data/TidyVoice/TidyVoiceX2_ASV"
TASK1_OUTPUT_TXT = "results/tl26_closed_lid.txt"

# Task 2: language verification
TASK2_ENROLL_TSV = "./eval_data/tl26_enroll.tsv"
TASK2_ENROLL_PATH = "data/TidyVoice/TidyVoiceX2_ASV"
TASK2_PAIRS_TXT = "./eval_data/tl26_pairs.txt"
TASK2_DATA_PATH = "data/TidyVoice/TidyVoiceX2_ASV"
TASK2_OUTPUT_TXT = "results/tl26_closed_pairs.txt"

# Ensure the output dirs for Task 1 / Task 2 exist
os.makedirs(os.path.dirname(TASK1_OUTPUT_TXT), exist_ok=True)
os.makedirs(os.path.dirname(TASK2_OUTPUT_TXT), exist_ok=True)

# The LANGUAGES list and its derived maps are no longer used once the dynamic
# label dictionary is adopted; kept only for backward compatibility.
LANGUAGES = [
    "ab", "ar", "ba", "be", "bg", "bn", "ca", "cv", "cy", "de",
    "dv", "el", "en", "fa", "fr", "ha", "hi", "hsb", "hy-AM", "ja",
    "ka", "lg", "lt", "mk", "ml", "mr", "nl", "or", "pl", "pt",
    "ru", "ta", "th", "tk", "tr", "ug", "uz", "yo", "yue", "zh-CN"
]

LABEL2ID = {lang: idx for idx, lang in enumerate(LANGUAGES)}
ID2LABEL = {idx: lang for idx, lang in enumerate(LANGUAGES)}
NUM_CLASSES = len(LANGUAGES)

TIDY_TO_VOX_MAPPING = {
    "zh-CN": "zh",
    "hy-AM": "hy",
}
