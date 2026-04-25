import cv2
import os
import sys

# ─────────────────────────────────────────
#  GENDER FACE CAPTURE TOOL
#  Chalaane ka tarika:
#    python run.py capture    →  faces capture karo
#    python run.py train      →  model train karo
#    python run.py predict    →  live prediction karo
# ─────────────────────────────────────────

DATASET_PATH = "dataset"
MODEL_PATH   = "model/gender_model.h5"
LABEL_MAP    = {0: "Male", 1: "Female"}
IMG_SIZE     = (200, 200)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ══════════════════════════════════════════
#  STEP 1 — CAPTURE
# ══════════════════════════════════════════
def capture():
    print("\n" + "="*50)
    print("  FACE CAPTURE TOOL")
    print("="*50)

    # Gender select karo
    print("\nKisaki photo leni hai?")
    print("  1 → Male")
    print("  2 → Female")
    choice = input("\nChoice dalo (1 ya 2): ").strip()

    if choice == "1":
        gender = "male"
        label_id = 0
    elif choice == "2":
        gender = "female"
        label_id = 1
    else:
        print("Galat choice! 1 ya 2 dalo.")
        return

    # Kitni photos?
    try:
        num = int(input(f"Kitni photos leni hain? (default 100): ").strip() or "100")
    except ValueError:
        num = 100

    save_path = os.path.join(DATASET_PATH, gender)
    os.makedirs(save_path, exist_ok=True)

    # Count existing images
    existing = len([f for f in os.listdir(save_path) if f.endswith(".jpg")])
    count = existing

    print(f"\n[INFO] Saving to: {save_path}/")
    print(f"[INFO] Already captured: {existing} images")
    print(f"[INFO] Will capture {num} more → Total: {existing + num}")
    print("\n'SPACE' dabao photo lene ke liye")
    print("'q' dabao band karne ke liye\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera nahi mili! Check karo.")
        return

    captured_now = 0

    while captured_now < num:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        face_found = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Status bar
        color = (0, 255, 0) if face_found else (0, 0, 255)
        status = f"Face: {'Detected' if face_found else 'Not Found'}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Gender: {gender.upper()}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.putText(display, f"Captured: {captured_now}/{num}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "SPACE=Save  Q=Quit", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Gender Capture Tool", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Band kar diya.")
            break

        elif key == ord(' '):
            if not face_found:
                print("[WARN] Koi face nahi mila! Pehle face frame mein lao.")
                continue

            # Sabse bada face lo (main subject)
            faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces_sorted[0]

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, IMG_SIZE)

            count += 1
            captured_now += 1
            filename = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[SAVED] {filename}  ({captured_now}/{num})")

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    male_count   = len(os.listdir(os.path.join(DATASET_PATH, "male")))   if os.path.exists(os.path.join(DATASET_PATH, "male"))   else 0
    female_count = len(os.listdir(os.path.join(DATASET_PATH, "female"))) if os.path.exists(os.path.join(DATASET_PATH, "female")) else 0

    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    print(f"  Male   images : {male_count}")
    print(f"  Female images : {female_count}")
    print(f"  Total         : {male_count + female_count}")
    print("\nAb 'python run.py train' chalaao model train karne ke liye.")


# ══════════════════════════════════════════
#  STEP 2 — TRAIN
# ══════════════════════════════════════════
def train():
    import numpy as np

    print("\n" + "="*50)
    print("  MODEL TRAINING")
    print("="*50)

    faces  = []
    labels = []

    for label_id, gender in LABEL_MAP.items():
        folder = os.path.join(DATASET_PATH, gender.lower())
        if not os.path.exists(folder):
            print(f"[WARN] Folder nahi mila: {folder}")
            continue

        imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        print(f"[INFO] {gender}: {len(imgs)} images mil gayi")

        for img_name in imgs:
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(label_id)

    if len(faces) < 2:
        print("\n[ERROR] Kum se kum 2 images chahiye! Pehle capture karo.")
        return

    print(f"\n[INFO] Total training images: {len(faces)}")
    print("[INFO] Training shuru ho raha hai... thoda wait karo.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(faces, np.array(labels))

    os.makedirs("model", exist_ok=True)
    recognizer.save(MODEL_PATH)

    print(f"\n[DONE] Model saved: {MODEL_PATH}")
    print("Ab 'python run.py predict' chalaao live prediction ke liye!")


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

    CONFIDENCE_THRESHOLD = 80  # Lower = better match

    print("\n" + "="*50)
    print("  LIVE GENDER PREDICTION")
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
            label_id, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                label = LABEL_MAP.get(label_id, "Unknown")
                conf_text = f"{100 - int(confidence)}% sure"
                color = (0, 200, 0) if label == "Male" else (200, 0, 200)
            else:
                label = "Unknown"
                conf_text = "Low confidence"
                color = (0, 0, 200)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, conf_text, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.imshow("Gender Prediction", frame)
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
║       GENDER RECOGNITION TOOL           ║
╠══════════════════════════════════════════╣
║  STEP 1:  python run.py capture          ║
║           → Webcam se face capture karo  ║
║           → Male ya Female select karo   ║
║                                          ║
║  STEP 2:  python run.py train            ║
║           → Dataset se model train karo  ║
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
