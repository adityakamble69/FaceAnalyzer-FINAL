# 🤖 Face Analyzer — AI Projects Collection

Yeh ek complete **Face AI System** hai jisme 4 alag tools hain:

| Project | Kya Karta Hai | Technology |
|---------|--------------|------------|
| 🎭 **Face Analyzer** | Age + Gender + Emotion ek saath | DeepFace + Claude API |
| 👴 **Age Recognition** | 5 age groups detect karta hai | OpenCV LBPH |
| ⚥ **Gender Recognition** | Male / Female detect karta hai | OpenCV LBPH |
| 😊 **Emotion Recognition** | 7 emotions detect karta hai | OpenCV LBPH |

---

## 🗂️ Poora Project Structure

```
FACE_ANALYZER - MAIN/
│
├── 📁 age_recognition/
│   └── age_recognition/
│       ├── dataset/          ← Training images (age groups)
│       ├── model/            ← Trained model save hoga
│       ├── README.md
│       └── run.py            ← Chalane wali file
│
├── 📁 emotion_recognition/
│   └── emotion_recognition/
│       ├── dataset/          ← Training images (emotions)
│       ├── model/            ← Trained model save hoga
│       └── run.py            ← Chalane wali file
│
├── 📁 gender_recognition/
│   └── gender_recognition/
│       ├── dataset/          ← Training images (male/female)
│       ├── model/            ← Trained model save hoga
│       ├── README.md
│       └── run.py            ← Chalane wali file
│
├── 📁 models/
│   ├── age_model.h5          ← DeepFace age model
│   ├── emotion_model.h5      ← DeepFace emotion model
│   └── gender_model.h5       ← DeepFace gender model
│
├── 📁 results/               ← Face Analyzer ke JSON results
│   └── result_YYYYMMDD_*.json
│
├── 📁 src/
│   ├── face_analyzer.py      ← Main GUI app
│   ├── api_client.py         ← Claude Vision API
│   ├── local_models.py       ← DeepFace wrapper
│   └── utils.py              ← Helper functions
│
├── .env                      ← API key yahan rakho
├── README.md                 ← Yahi file hai
└── requirements.txt          ← Saari dependencies
```

---

## ⚙️ Ek Baar Install Karo

### Project 1, 2, 3 ke liye (Age / Gender / Emotion Recognition):
```bash
pip install opencv-python opencv-contrib-python numpy
```

### Project 4 ke liye (Face Analyzer GUI):
```bash
pip install deepface tf-keras opencv-python pillow numpy anthropic
```

Ya seedha requirements.txt se:
```bash
pip install -r requirements.txt
```

---
---

# 📘 Project 1 — Face Analyzer (Main GUI App)

**Kya hai:** Ek complete desktop GUI app jo ek saath **Age + Gender + Emotion** detect karta hai. Camera ya photo upload — dono se kaam karta hai.

**Do modes hain:**
- 🔵 **Local Mode** — `.h5` models use karta hai, internet nahi chahiye
- 🟣 **API Mode** — Claude Vision API use karta hai, zyada accurate

### Folder
```
FACE_ANALYZER - MAIN/
└── src/
    └── face_analyzer.py   ← Yahi chalao
```

### Chalane Se Pehle (API Mode ke liye)
`.env` file mein apni API key daalo:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

### Chalao
```bash
cd src
python face_analyzer.py
```

### App Kaise Use Karo
1. **Camera** button dabao → webcam on hoga
2. Ya **Upload** button dabao → photo select karo
3. **Analyze Karo** button dabao
4. Results right side mein dikhenge:
   - 👴 Umar (Age + Range)
   - ⚥ Gender
   - 😊 Dominant Emotion
   - 📊 Emotion breakdown bars
5. **Result Save Karo** → `results/` folder mein JSON save hoga

### Models Kahan Rakho
```
FACE_ANALYZER - MAIN/
└── models/
    ├── age_model.h5
    ├── gender_model.h5
    └── emotion_model.h5
```
> Agar models nahi hain toh DeepFace automatically download kar lega pehli baar.

---
---

# 📗 Project 2 — Age Recognition

**Kya hai:** Webcam se insaan ka **age group** detect karta hai. Custom dataset se train hota hai.

### 5 Age Groups

| Choice | Group | Age Range |
|--------|-------|-----------|
| 0 | 👶 Child | 0 – 12 |
| 1 | 🧒 Teen | 13 – 19 |
| 2 | 🧑 Young Adult | 20 – 35 |
| 3 | 🧔 Middle Age | 36 – 55 |
| 4 | 👴 Senior | 56+ |

### Folder
```
age_recognition/
└── age_recognition/
    └── run.py   ← Yahi chalao
```

### Step 1 — Dataset Capture Karo
```bash
cd age_recognition/age_recognition
python run.py capture
```
- Age group number dalo (0-4)
- Kitni photos chahiye batao (default: 100)
- **`SPACE`** dabao → photo save hogi
- **`Q`** dabao → band karo
- **Yeh step baar baar karo** — har age group ke liye alag alag

### Step 2 — Model Train Karo
```bash
python run.py train
```
> `model/age_model.yml` ban jayega

### Step 3 — Live Predict Karo
```bash
python run.py predict
```

---
---

# 📙 Project 3 — Gender Recognition

**Kya hai:** Webcam se **Male ya Female** detect karta hai. Custom dataset se train hota hai.

### 2 Categories

| Choice | Label |
|--------|-------|
| 1 | 👨 Male |
| 2 | 👩 Female |

### Folder
```
gender_recognition/
└── gender_recognition/
    └── run.py   ← Yahi chalao
```

### Step 1 — Dataset Capture Karo
```bash
cd gender_recognition/gender_recognition
python run.py capture
```
- `1` (Male) ya `2` (Female) dalo
- Kitni photos chahiye batao (default: 100)
- **`SPACE`** dabao → photo save hogi
- **`Q`** dabao → band karo
- Male ke baad Female ke liye dobara chalao

### Step 2 — Model Train Karo
```bash
python run.py train
```
> `model/gender_model.yml` ban jayega

### Step 3 — Live Predict Karo
```bash
python run.py predict
```

---
---

# 📕 Project 4 — Emotion Recognition

**Kya hai:** Webcam se real-time **7 emotions** detect karta hai. Custom dataset se train hota hai.

### 7 Emotions

| Choice | Emotion | Matlab |
|--------|---------|--------|
| 0 | 😠 Angry | Gussa |
| 1 | 🤢 Disgust | Nafrat |
| 2 | 😨 Fear | Dar |
| 3 | 😊 Happy | Khushi |
| 4 | 😢 Sad | Udaasi |
| 5 | 😮 Surprise | Hairani |
| 6 | 😐 Neutral | Normal |

### Folder
```
emotion_recognition/
└── emotion_recognition/
    └── run.py   ← Yahi chalao
```

### Step 1 — Dataset Capture Karo
```bash
cd emotion_recognition/emotion_recognition
python run.py capture
```
- Emotion number dalo (0-6)
- Kitni photos chahiye batao (default: 100)
- **`SPACE`** dabao → photo save hogi
- **`Q`** dabao → band karo
- **7 baar karo** — har emotion ke liye alag alag

### Step 2 — Model Train Karo
```bash
python run.py train
```
> `model/emotion_model.yml` ban jayega

### Step 3 — Live Predict Karo
```bash
python run.py predict
```

---
---

## 💡 Dataset Tips — Sabhi Projects Ke Liye

Jitna achha dataset, utna achha model!

- 📸 **Minimum 50-100 images** per category lo
- 🔆 **Alag alag lighting** mein lo — subah, raat, lamp ke saath
- 🔄 **Alag angles** — seedha, left, right, thoda neeche
- 👥 **Zyada log** — sirf apni nahi, doston ki bhi photos lo (unki permission se)
- 😠 **Clear expression** — emotion capture karte waqt exaggerate karo

---

## ❓ Common Errors Aur Fix

| Error | Fix |
|-------|-----|
| `can't open file 'run.py'` | Sahi folder mein jao: `cd project_name/project_name` |
| `Camera nahi mili` | Doosra app camera use kar raha hoga — band karo |
| `Bahut kum images` | Pehle `capture` karo, phir `train` |
| `Sirf 1 class hai` | Kam se kam 2 categories ke liye capture karo |
| `deepface install error` | `pip install deepface tf-keras` chalao |
| `No module named cv2` | `pip install opencv-python opencv-contrib-python` |
| `API key error` | `.env` file mein `ANTHROPIC_API_KEY=sk-ant-...` daalo |

---

## 🔁 Quick Reference — Sabhi Commands

```bash
# ── Age Recognition ──────────────────────────────
cd age_recognition/age_recognition
python run.py capture      # Dataset lo
python run.py train        # Model banao
python run.py predict      # Test karo

# ── Gender Recognition ───────────────────────────
cd gender_recognition/gender_recognition
python run.py capture
python run.py train
python run.py predict

# ── Emotion Recognition ──────────────────────────
cd emotion_recognition/emotion_recognition
python run.py capture
python run.py train
python run.py predict

# ── Face Analyzer (Main GUI) ─────────────────────
cd src
python face_analyzer.py
```

---

*100% Offline (Local Models) — Internet sirf API mode ke liye chahiye* 🔒
