"""
local_models.py
===============
DeepFace ke official models use karta hai jo tumhare paas hain.

Pehle yeh karo — models sahi jagah daalo:
  Windows:  C:\\Users\\<tumhara_naam>\\.deepface\\weights\\
  Mac/Linux: ~/.deepface/weights/

Wahan yeh files honi chahiye:
  - age_model_weights.h5
  - gender_model_weights.h5
  - facial_expression_model_weights.h5

Agar tumhare files ka naam alag hai (age_model.h5) toh
neeche RENAME_MAP mein sahi naam likh do.
"""

import os
import shutil
import numpy as np
import cv2

# ── DeepFace weights folder ───────────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(os.path.expanduser("~"), ".deepface", "weights")

# Agar tumhare models ka naam alag hai toh yahan map karo:
# { "tumhari_file.h5" : "deepface_expected_name.h5" }
RENAME_MAP = {
    "age_model.h5":     "age_model_weights.h5",
    "gender_model.h5":  "gender_model.h5",
    "emotion_model.h5": "facial_expression_model_weights.h5",
}

# Tumhare models ka folder (face_analyzer/models/)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


def _copy_models_to_deepface():
    """
    Tumhare models/ folder se files ~/.deepface/weights/ mein copy karo
    agar wahan nahi hain.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for src_name, dst_name in RENAME_MAP.items():
        src = os.path.join(MODELS_DIR, src_name)
        dst = os.path.join(WEIGHTS_DIR, dst_name)

        if os.path.exists(src) and not os.path.exists(dst):
            print(f"[Setup] {src_name} → {dst_name} copy ho raha hai...")
            shutil.copy2(src, dst)
            print(f"[Setup] Done: {dst}")
        elif os.path.exists(dst):
            print(f"[OK] {dst_name} already hai")
        else:
            print(f"[Warning] {src_name} nahi mila models/ mein")


# Models copy karo (sirf ek baar)
_copy_models_to_deepface()

# Ab DeepFace import karo
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False
    print("[Error] deepface install karo: pip install deepface tf-keras")

# Face detector
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def analyze_local(frame: np.ndarray) -> dict:
    """
    OpenCV BGR frame leke face detect karo aur DeepFace se analyze karo.

    Returns:
        dict with age, gender, dominant_emotion, emotions
    """
    if not DEEPFACE_OK:
        return {"error": "deepface install karo: pip install deepface tf-keras"}

    # Face detect karo pehle
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return {"error": "Koi chehra nahi mila — thoda aur seedha dekho!"}

    # Sabse bada face crop karo
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop   = frame[y:y+h, x:x+w]

    try:
        result = DeepFace.analyze(
            img_path         = face_crop,
            actions          = ["age", "gender", "emotion"],
            enforce_detection= False,
            silent           = True,
        )

        # List ya dict dono handle karo
        r = result[0] if isinstance(result, list) else result

        # Gender
        gender_raw = r.get("dominant_gender", r.get("gender", "Unknown"))
        if isinstance(gender_raw, dict):
            gender = max(gender_raw, key=gender_raw.get)
        else:
            gender = str(gender_raw)

        age      = int(r.get("age", 0))
        emotions = {k: round(float(v), 1) for k, v in r.get("emotion", {}).items()}
        dominant = r.get("dominant_emotion", "neutral")

        return {
            "age":              age,
            "age_range":        f"{max(1, age-4)}–{age+4}",
            "gender":           gender.capitalize(),
            "dominant_emotion": dominant,
            "emotions":         emotions,
        }

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)[:100]}"}
