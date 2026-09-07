"""
config.py
=========
Semua konfigurasi & placeholder yang WAJIB kamu cek/sesuaikan ada di sini.
Cari komentar "TODO" untuk bagian yang harus kamu ubah sesuai model kamu.
"""

import torch

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# CHECKPOINT PATHS
# TODO: ganti dengan path file .pt / .pth hasil training kamu.
# Letakkan file model di folder checkpoints/ (sudah disediakan foldernya).
# ---------------------------------------------------------------------------
ETHNICITY_MODEL_PATH = "efficientnet_b3"  # TODO: ganti sesuai model yang kamu pakai
ETHNICITY_CHECKPOINT = {
    "resnet50" : "checkpoints/model_resnet50_mulut_BEST.pth",
    "efficientnet_b3" : "checkpoints/model_efficientnetb3_p7_BEST.pth",
    "convnext_tiny" : "checkpoints/model_convnext_tiny_p3_BEST.pth",
    "vit_b16" : "checkpoints/model_vitb16_p1_BEST.pth",
    }

ETHNICITY_CHECKPOINT_PATH = ETHNICITY_CHECKPOINT.get(ETHNICITY_MODEL_PATH, ETHNICITY_CHECKPOINT["efficientnet_b3"])
EMOTION_CHECKPOINT_PATH = "checkpoints/best_efficientnet_fer1200.pth.zip"

# ---------------------------------------------------------------------------
# ARCHITEKTUR MODEL
# TODO: sesuaikan varian EfficientNet dengan yang kamu pakai waktu training
# (b0, b1, b2, b3, b4 ...). Default: efficientnet_b0.
# ---------------------------------------------------------------------------
EFFICIENTNET_VARIANT = "efficientnet_b0"

# ---------------------------------------------------------------------------
# CLASS LABELS
# TODO (PENTING): urutan list ini HARUS SAMA PERSIS dengan urutan index kelas
# waktu training (biasanya urutan alfabetis nama folder dataset kamu, kalau
# pakai ImageFolder). Kalau urutannya salah, prediksi akan ketuker labelnya.
# Ini baru PLACEHOLDER, ganti sebelum dipakai demo asli!
# ---------------------------------------------------------------------------
ETHNICITY_CLASSES = [
    "Ambon",     # index 0  -- TODO: cek urutan asli
    "Batak",    # index 1
    "Chinese",    # index 2
    "Jawa",   # index 3
    "Kalimantan",   # index 4
    "Papua",    # index 5
    "Toraja",    # index 6
]

# Urutan umum dataset FER (silakan cocokkan dengan urutan training kamu)
EMOTION_CLASSES = [
    "Angry",     # index 0 -- TODO: cek urutan asli
    "Disgust",   # index 1
    "Fear",      # index 2
    "Happy",     # index 3
    "Sad",       # index 4
    "Surprise",  # index 5
]

NUM_ETHNICITY_CLASSES = len(ETHNICITY_CLASSES)
NUM_EMOTION_CLASSES = len(EMOTION_CLASSES)

# ---------------------------------------------------------------------------
# PREPROCESSING
# TODO: sesuaikan image size & normalisasi dengan pipeline training kamu.
# Default di bawah ini pakai standar ImageNet (umum dipakai kalau kamu
# fine-tune dari pretrained torchvision).
# ---------------------------------------------------------------------------
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# FACE DETECTION (MediaPipe)
# ---------------------------------------------------------------------------
FACE_DETECTION_MIN_CONFIDENCE = 0.5
FACE_DETECTION_MODEL_SELECTION = 1  # 0 = jarak dekat (<2m), 1 = jarak jauh, cocok utk demo di panggung
FACE_CROP_PADDING_RATIO = 0.20      # tambahan padding di sekitar bbox wajah sebelum di-crop

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
APP_TITLE = "RupaNusantara AI"
APP_SUBTITLE = "Gelar Inovasi Harmoni (GIHN)"
