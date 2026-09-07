"""
models.py
=========
Definisi arsitektur, loading checkpoint, dan fungsi inference
untuk model etnis (ResNet50) dan model emosi (EfficientNet).
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import config


class ModelLoadError(Exception):
    """Dilempar kalau checkpoint tidak ditemukan / gagal dimuat."""
    pass


# ---------------------------------------------------------------------------
# Preprocessing transform (dipakai bersama oleh kedua model)
# TODO: kalau preprocessing training kamu berbeda (mis. tidak pakai
# normalisasi ImageNet, atau ukuran input beda), sesuaikan di config.py
# ---------------------------------------------------------------------------
_preprocess = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
])


def preprocess_face(face_rgb_ndarray):
    """Ubah crop wajah (numpy array RGB, HxWx3) jadi tensor siap masuk model."""
    pil_img = Image.fromarray(face_rgb_ndarray)
    tensor = _preprocess(pil_img)
    return tensor.unsqueeze(0)  # tambah batch dimension -> (1, 3, H, W)


def _build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 7)
    )
    return model


def _build_efficientnet(variant, num_classes):
    if not hasattr(models, variant):
        raise ModelLoadError(
            f"Varian EfficientNet '{variant}' tidak dikenal di torchvision. "
            f"Cek config.EFFICIENTNET_VARIANT (contoh valid: efficientnet_b0, "
            f"efficientnet_b1, dst)."
        )
    constructor = getattr(models, variant)
    model = constructor(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _load_state_dict_flexible(model, checkpoint_path, device):
    """
    Loading checkpoint yang cukup fleksibel:
    - mendukung file yang isinya langsung state_dict
    - atau dict dengan key umum seperti 'state_dict' / 'model_state_dict'
    """
    if not os.path.isfile(checkpoint_path):
        raise ModelLoadError(
            f"File checkpoint tidak ditemukan: '{checkpoint_path}'.\n"
            f"Letakkan file .pt/.pth hasil training kamu di path tersebut "
            f"(cek config.py)."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ModelLoadError(f"Gagal membaca checkpoint '{checkpoint_path}': {e}")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # buang prefix "module." kalau checkpoint disimpan dari DataParallel
    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except Exception as e:
        raise ModelLoadError(
            f"State dict tidak cocok dengan arsitektur model untuk "
            f"'{checkpoint_path}'. Kemungkinan arsitektur/jumlah kelas beda "
            f"dengan config.py. Detail error: {e}"
        )

    return model


def load_ethnicity_model():
    """
    Memuat model secara fleksibel berdasarkan config.ETHNICITY_MODEL_TYPE
    hanya untuk model yang dipilih saja
    """
    model_type = getattr(config, "ETHNICITY_MODEL_TYPE", "efficientnet_b3")
    ckpt_path = config.ETHNICITY_CHECKPOINT_PATH
    num_classes = config.NUM_ETHNICITY_CLASSES
    device = config.DEVICE
    
    try:
        if model_type == "resnet50":
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif model_type == "efficientnet_b3":
            model = models.efficientnet_b3(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            
        elif model_type == "convnext_tiny":
            model = models.convnext_tiny(weights=None)
            num_ftrs = model.classifier[2].in_features
            model.classifier[2] = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif model_type == "vit_b16":
            model = models.vit_b_16(weights=None)
            num_ftrs = model.heads.head.in_features 
            model.heads.head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            
        else:
            raise ValueError(f"Tipe model etnis '{model_type}' tidak dikenali!")
        
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        return model
    
    except Exception as e:
        raise ModelLoadError(f"Gagal memuat arsitektur {model_type} dari {ckpt_path}: {e}")


def load_emotion_model():
    model = _build_efficientnet(config.EFFICIENTNET_VARIANT, config.NUM_EMOTION_CLASSES)
    model = _load_state_dict_flexible(model, config.EMOTION_CHECKPOINT_PATH, config.DEVICE)
    model.to(config.DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, face_rgb_ndarray, class_names):
    """
    Jalankan satu model pada satu crop wajah.
    Return: (label:str, confidence:float, semua_probabilitas:dict)
    """
    tensor = preprocess_face(face_rgb_ndarray).to(config.DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    label = class_names[top_idx]
    confidence = float(probs[top_idx])
    all_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return label, confidence, all_probs
