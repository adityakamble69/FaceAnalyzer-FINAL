# Gender Recognition Tool

## Pehle install karo
```
pip install opencv-python opencv-contrib-python numpy
```

## Chalaane ka tarika

### Step 1 — Faces Capture Karo
```
python run.py capture
```
- Camera on hoga
- Male ya Female select karo
- **SPACE** dabao → face save hoga
- **Q** dabao → band karo
- Baar baar chalao alag logon ke liye

### Step 2 — Model Train Karo
```
python run.py train
```
- Dataset se automatically model train hoga
- `model/gender_model.yml` mein save hoga

### Step 3 — Live Predict Karo
```
python run.py predict
```
- Camera pe face dikhao
- Real-time Male/Female prediction

---

## Project Structure
```
gender_recognition/
├── run.py              ← main file
├── dataset/
│   ├── male/           ← male faces (.jpg)
│   └── female/         ← female faces (.jpg)
└── model/
    └── gender_model.yml ← trained model
```

## Tips
- Minimum **50-100 images per gender** → better accuracy
- Alag alag lighting mein capture karo
- Side angles bhi capture karo
- Glasses, hat, etc. ke saath bhi kuch photos lo
