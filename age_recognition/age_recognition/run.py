import cv2
import os
import sys

# ─────────────────────────────────────────
#  AGE GROUP FACE CAPTURE & RECOGNITION
#  Chalaane ka tarika:
#    python run.py capture    →  faces capture karo
#    python run.py train      →  model train karo
#    python run.py predict    →  live prediction karo
# ─────────────────────────────────────────

DATASET_PATH = "dataset"
MODEL_PATH   = "model/age_model.h5"
IMG_SIZE     = (200, 200)

AGE_GROUPS = {
    0: "Child (0-12)",
    1: "Teen (13-19)",
    2: "Young Adult (20-35)",
    3: "Middle Age (36-55)",
    4: "Senior (56+)",
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ══════════════════════════════════════════
#  STEP 1 — CAPTURE
# ══════════════════════════════════════════
def capture():
    print("\n" + "="*50)
    print("  AGE GROUP FACE CAPTURE TOOL")
    print("="*50)

    print("\nKis age group ki photo leni hai?")
    for key, val in AGE_GROUPS.items():
        print(f"  {key} → {val}")

    choice = input("\nChoice dalo (0-4): ").strip()

    if not choice.isdigit() or int(choice) not in AGE_GROUPS:
        print("Galat choice! 0 se 4 ke beech kuch dalo.")
        return

    label_id  = int(choice)
    age_label = AGE_GROUPS[label_id]
    folder_name = age_label.split("(")[0].strip().lower().replace(" ", "_")

    try:
        num = int(input(f"Kitni photos leni hain? (default 100): ").strip() or "100")
    except ValueError:
        num = 100

    save_path = os.path.join(DATASET_PATH, folder_name)
    os.makedirs(save_path, exist_ok=True)

    existing = len([f for f in os.listdir(save_path) if f.endswith(".jpg")])
    count = existing

    print(f"\n[INFO] Age group  : {age_label}")
    print(f"[INFO] Saving to  : {save_path}/")
    print(f"[INFO] Already captured : {existing} images")
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

        display   = frame.copy()
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces     = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        face_found = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        color  = (0, 255, 0) if face_found else (0, 0, 255)
        status = f"Face: {'Detected' if face_found else 'Not Found'}"
        cv2.putText(display, status,              (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Age: {age_label}", (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(display, f"Captured: {captured_now}/{num}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, "SPACE=Save  Q=Quit", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("Age Capture Tool", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Band kar diya.")
            break

        elif key == ord(' '):
            if not face_found:
                print("[WARN] Koi face nahi mila! Pehle face frame mein lao.")
                continue

            faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces_sorted[0]

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, IMG_SIZE)

            count         += 1
            captured_now  += 1
            filename = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[SAVED] {filename}  ({captured_now}/{num})")

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    total = 0
    for label_id, age_label in AGE_GROUPS.items():
        fname = age_label.split("(")[0].strip().lower().replace(" ", "_")
        fpath = os.path.join(DATASET_PATH, fname)
        count_imgs = len([f for f in os.listdir(fpath) if f.endswith(".jpg")]) if os.path.exists(fpath) else 0
        total += count_imgs
        print(f"  {age_label:<22} : {count_imgs} images")
    print(f"  {'Total':<22} : {total} images")
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

    for label_id, age_label in AGE_GROUPS.items():
        folder_name = age_label.split("(")[0].strip().lower().replace(" ", "_")
        folder = os.path.join(DATASET_PATH, folder_name)

        if not os.path.exists(folder):
            print(f"[SKIP] Folder nahi mila: {folder}")
            continue

        imgs = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        if len(imgs) == 0:
            print(f"[SKIP] {age_label}: koi image nahi mili")
            continue

        print(f"[INFO] {age_label:<22} : {len(imgs)} images")

        for img_name in imgs:
            img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(label_id)

    if len(faces) < 2:
        print("\n[ERROR] Kum se kum 2 images chahiye training ke liye!")
        print("Pehle 'python run.py capture' se images lo.")
        return

    # Minimum 2 unique classes check
    unique_classes = set(labels)
    if len(unique_classes) < 2:
        print(f"\n[ERROR] Sirf ek age group ({AGE_GROUPS[list(unique_classes)[0]]}) ki images hain.")
        print("Training ke liye kum se kum 2 alag age groups ki images chahiye.")
        return

    print(f"\n[INFO] Total images   : {len(faces)}")
    print(f"[INFO] Age groups     : {len(unique_classes)}")
    print("[INFO] Training shuru ho raha hai...")

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

    CONFIDENCE_THRESHOLD = 85

    # Age group ke liye alag colors
    COLORS = {
        0: (255, 165,   0),   # Orange  — Child
        1: (  0, 255, 255),   # Cyan    — Teen
        2: (  0, 255,   0),   # Green   — Young Adult
        3: (255,   0, 255),   # Magenta — Middle Age
        4: (  0, 165, 255),   # Blue    — Senior
    }

    print("\n" + "="*50)
    print("  LIVE AGE GROUP PREDICTION")
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
            face_roi  = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
            label_id, confidence = recognizer.predict(face_roi)

            if confidence < CONFIDENCE_THRESHOLD:
                label      = AGE_GROUPS.get(label_id, "Unknown")
                conf_text  = f"{max(0, 100 - int(confidence))}% sure"
                color      = COLORS.get(label_id, (0, 255, 0))
            else:
                label      = "Unknown"
                conf_text  = "Low confidence"
                color      = (0, 0, 200)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label,     (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame, conf_text, (x, y -  8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.imshow("Age Group Prediction", frame)
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
║        AGE GROUP RECOGNITION TOOL       ║
╠══════════════════════════════════════════╣
║  Age Groups:                             ║
║    0 → Child       (0-12)                ║
║    1 → Teen        (13-19)               ║
║    2 → Young Adult (20-35)               ║
║    3 → Middle Age  (36-55)               ║
║    4 → Senior      (56+)                 ║
╠══════════════════════════════════════════╣
║  STEP 1:  python run.py capture          ║
║           → Age group select karo        ║
║           → SPACE se photos lo           ║
║                                          ║
║  STEP 2:  python run.py train            ║
║           → Model train karo             ║
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
