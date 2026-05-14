import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0003
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-4
UNFREEZE_LAYER4 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Recognition / Matching =====
MATCH_THRESHOLD = 0.35
MERGE_UNKNOWN_THRESHOLD = 0.45
DUPLICATE_SIM_THRESHOLD = 0.97

MIN_FACE_SIZE = 50
BLUR_THRESHOLD = 25.0

TRACK_TIMEOUT_SEC = 4.0
TRACK_MAX_CENTER_DISTANCE_RATIO = 0.9
TRACK_MIN_IOU_FOR_MATCH = 0.15

TRACK_CONFIRMATION_SAMPLES = 3
TRACK_MAX_PENDING_SAMPLES = 4
TRACK_CAPTURE_INTERVAL_SEC = 0.6
TRACK_EVENT_COOLDOWN_SEC = 8.0

PERSON_SAVE_COOLDOWN_SEC = 2.0
DETECTOR_BACKEND = "insightface"
DETECTOR_SIZE = 320
DEBUG_LOGS = False
SOURCE_NAME = "webcam0"
# ===== Output paths =====
OUT_ROOT = BASE_DIR / "outputs" / "collections"
DB_PATH = BASE_DIR / "outputs" / "collections_db.json"
INCIDENTS_PATH = BASE_DIR / "outputs" / "incidents.jsonl"
