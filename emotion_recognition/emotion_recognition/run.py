"""
Emotion Recognition Tool
========================
Chalaane ka tarika:
  python run.py capture    →  webcam se faces capture karo
  python run.py train      →  model train karo
  python run.py predict    →  live prediction karo
"""

import cv2
import os
import sys
import numpy as np

DATASET_PATH = "dataset"
MODEL_PATH   = "model/emotion_model.yml"
IMG_SIZE     = (200, 200)

EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

EMO_COLORS = {
    "Angry":    (0,   0,   255),
    "Disgust":  (0,   128, 0  ),
    "Fear":     (128, 0,   128),
    "Happy":    (0,   255, 128),
    "Sad":      (255, 100, 0  ),
    "Surprise": (0,   200, 255),
    "Neutral":  (160, 160, 160),
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ══════════════════════════════════════════
#  STEP 1 — CAPTURE
# ══════════════════════════════════════════
def capture():
    print("\n" + "="*50)
    print("  EMOTION FACE CAPTURE")
    print("="*50)

    print("\nKis emotion ki photo leni hai?")
    for key, val in EMOTIONS.items():
        folder = os.path.join(DATASET_PATH, val.lower())
        count  = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) if os.path.exists(folder) else 0
        print(f"  {key} → {val:<10}  (abhi: {count} images)")

    choice = input("\nChoice dalo (0-6): ").strip()
    if not choice.isdigit() or int(choice) not in EMOTIONS:
        print("Galat choice! 0 se 6 ke beech dalo.")
        return

    label_id    = int(choice)
    emotion     = EMOTIONS[label_id]
    save_path   = os.path.join(DATASET_PATH, emotion.lower())
    os.makedirs(save_path, exist_ok=True)

    try:
        num = int(input(f"Kitni photos? (default 100): ").strip() or "100")
    except ValueError:
        num = 100

    existing     = len([f for f in os.listdir(save_path) if f.endswith(".jpg")])
    count        = existing
    captured_now = 0

    print(f"\n[INFO] Emotion   : {emotion.upper()}")
    print(f"[INFO] Saving to : {save_path}/")
    print(f"[INFO] Already   : {existing} images")
    print(f"[INFO] Target    : {num} more")
    print("\n'SPACE' dabao face save karne ke liye")
    print("'q' dabao band karne ke liye\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera nahi mili!")
        return

    while captured_now < num:
        ret, frame = cap.read()
        if not ret:
            break

        display    = frame.copy()
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces      = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_found = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        color  = (0, 255, 0) if face_found else (0, 0, 255)
        status = "Face: Detected" if face_found else "Face: Not Found"
        cv2.putText(display, status,                         (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Emotion: {emotion.upper()}",  (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(display, f"Captured: {captured_now}/{num}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, "SPACE=Save  Q=Quit",            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Emotion Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Band kar diya.")
            break
        elif key == ord(' '):
            if not face_found:
                print("[WARN] Koi face nahi mila!")
                continue

            faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h   = faces_sorted[0]
            face_img      = gray[y:y+h, x:x+w]
            face_img      = cv2.resize(face_img, IMG_SIZE)

            count        += 1
            captured_now += 1
            fname = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(fname, face_img)
            print(f"[SAVED] {fname}  ({captured_now}/{num})")

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    total = 0
    for label_id, emo in EMOTIONS.items():
        folder = os.path.join(DATASET_PATH, emo.lower())
        n = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) if os.path.exists(folder) else 0
        total += n
        bar = "█" * (n // 5)
        print(f"  {emo:<10} : {n:>4} images  {bar}")
    print(f"\n  Total      : {total} images")
    print("\nAb 'python run.py train' chalao!")


# ══════════════════════════════════════════
#  STEP 2 — TRAIN
# ══════════════════════════════════════════
def train():
    print("\n" + "="*50)
    print("  MODEL TRAINING")
    print("="*50)

    faces  = []
    labels = []

    for label_id, emotion in EMOTIONS.items():
        folder = os.path.join(DATASET_PATH, emotion.lower())
        if not os.path.exists(folder):
            print(f"[SKIP] {emotion}: folder nahi mila")
            continue

        imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        if not imgs:
            print(f"[SKIP] {emotion}: koi image nahi")
            continue

        print(f"[INFO] {emotion:<10} : {len(imgs)} images")
        for img_name in imgs:
            img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(label_id)

    if len(faces) < 2:
        print("\n[ERROR] Bahut kum images! Pehle capture karo.")
        return

    unique = set(labels)
    if len(unique) < 2:
        print(f"\n[ERROR] Sirf 1 emotion class hai. Kam se kam 2 chahiye!")
        return

    print(f"\n[INFO] Total images : {len(faces)}")
    print(f"[INFO] Emotions     : {len(unique)}")
    print("[INFO] Training ho raha hai...")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))

    os.makedirs("model", exist_ok=True)
    recognizer.save(MODEL_PATH)

    print(f"\n[DONE] Model saved: {MODEL_PATH}")
    print("Ab 'python run.py predict' chalao!")


# ══════════════════════════════════════════
#  STEP 3 — PREDICT
# ══════════════════════════════════════════
def predict():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model nahi mila: {MODEL_PATH}")
        print("Pehle 'python run.py train' chalao!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    CONFIDENCE_THRESHOLD = 85

    print("\n" + "="*50)
    print("  LIVE EMOTION PREDICTION")
    print("="*50)
    print("Camera on ho raha hai... 'q' dabao band karne ke liye.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera nahi mili!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi         = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
            label_id, conf   = recognizer.predict(face_roi)

            if conf < CONFIDENCE_THRESHOLD:
                emotion   = EMOTIONS.get(label_id, "Unknown")
                conf_text = f"{max(0, 100 - int(conf))}% sure"
                color     = EMO_COLORS.get(emotion, (0, 255, 0))
            else:
                emotion   = "Unknown"
                conf_text = "Low confidence"
                color     = (100, 100, 100)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion,   (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, conf_text, (x, y -  8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.imshow("Emotion Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Band ho gaya.")


# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════╗
║      EMOTION RECOGNITION TOOL           ║
╠══════════════════════════════════════════╣
║  Emotions:                               ║
║    0 → Angry      3 → Happy              ║
║    1 → Disgust    4 → Sad                ║
║    2 → Fear       5 → Surprise           ║
║                   6 → Neutral            ║
╠══════════════════════════════════════════╣
║  STEP 1:  python run.py capture          ║
║           → Har emotion ke liye faces lo ║
║           → SPACE se photo save karo     ║
║                                          ║
║  STEP 2:  python run.py train            ║
║           → Model train hoga             ║
║                                          ║
║  STEP 3:  python run.py predict          ║
║           → Live camera pe test karo     ║
╚══════════════════════════════════════════╝

Install karo pehle:
  pip install opencv-python opencv-contrib-python numpy
        """)
    elif sys.argv[1] == "capture":
        capture()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "predict":
        predict()
    else:
        print(f"[ERROR] '{sys.argv[1]}' pehchana nahi. Use: capture / train / predict")
