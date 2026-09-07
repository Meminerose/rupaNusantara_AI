"""
face_utils.py
=============
Deteksi banyak wajah sekaligus dalam satu frame/gambar menggunakan
MediaPipe Face Detection, lalu crop tiap wajah untuk diproses model.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

import config

mp_face_detection = mp.solutions.face_detection


@dataclass
class DetectedFace:
    box: tuple          # (x1, y1, x2, y2) dalam koordinat pixel, sudah di-clip ke batas gambar
    crop_rgb: np.ndarray  # crop wajah, format RGB, siap dipreprocess ke model
    detection_confidence: float


def get_face_detector():
    """
    Buat instance MediaPipe FaceDetection.
    Panggil .close() saat sudah tidak dipakai, atau gunakan sebagai context manager.
    """
    return mp_face_detection.FaceDetection(
        model_selection=config.FACE_DETECTION_MODEL_SELECTION,
        min_detection_confidence=config.FACE_DETECTION_MIN_CONFIDENCE,
    )


def detect_faces(detector, frame_rgb: np.ndarray):
    """
    frame_rgb: ndarray RGB (H, W, 3)
    Return: list[DetectedFace], bisa lebih dari satu wajah.
    """
    h, w = frame_rgb.shape[:2]
    results = detector.process(frame_rgb)

    faces = []
    if not results.detections:
        return faces

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        conf = detection.score[0] if detection.score else 0.0

        # koordinat relatif (0-1) -> pixel
        x1 = bbox.xmin * w
        y1 = bbox.ymin * h
        box_w = bbox.width * w
        box_h = bbox.height * h
        x2 = x1 + box_w
        y2 = y1 + box_h

        # tambah padding di sekitar wajah biar konteks (dagu/dahi) ikut kecrop
        pad_x = box_w * config.FACE_CROP_PADDING_RATIO
        pad_y = box_h * config.FACE_CROP_PADDING_RATIO
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

        # clip ke batas gambar & pastikan integer
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(w, x2))
        y2 = int(min(h, y2))

        if x2 <= x1 or y2 <= y1:
            continue  # bbox tidak valid, skip

        crop_rgb = frame_rgb[y1:y2, x1:x2].copy()
        if crop_rgb.size == 0:
            continue

        faces.append(DetectedFace(
            box=(x1, y1, x2, y2),
            crop_rgb=crop_rgb,
            detection_confidence=float(conf),
        ))

    return faces


def draw_annotations(frame_bgr: np.ndarray, face_box, ethnicity_text: str, emotion_text: str):
    """
    Gambar bounding box + label di atas frame BGR (format OpenCV) secara in-place.
    Dipanggil per wajah yang terdeteksi.
    """
    x1, y1, x2, y2 = face_box
    color = (0, 200, 255)  # BGR - oranye kekuningan, kontras di banyak latar

    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    label_line1 = f"Etnis: {ethnicity_text}"
    label_line2 = f"Emosi: {emotion_text}"

    # background hitam tipis di belakang teks biar tetap terbaca
    (tw1, th1), _ = cv2.getTextSize(label_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    (tw2, th2), _ = cv2.getTextSize(label_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    box_w = max(tw1, tw2) + 10
    text_y_top = max(0, y1 - (th1 + th2 + 20))

    cv2.rectangle(frame_bgr, (x1, text_y_top), (x1 + box_w, y1), (0, 0, 0), -1)
    cv2.putText(frame_bgr, label_line1, (x1 + 5, text_y_top + th1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, label_line2, (x1 + 5, text_y_top + th1 + th2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return frame_bgr
