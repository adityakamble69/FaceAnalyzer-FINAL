# Age Group Recognition Tool

## Pehle install karo
```
pip install opencv-python opencv-contrib-python numpy
```

## Chalaane ka tarika

### Step 1 — Faces Capture Karo
```
python run.py capture
```
- Age group select karo (0 se 4)
- **SPACE** dabao → face save hoga
- **Q** dabao → band karo
- Har age group ke liye alag alag chalao

### Step 2 — Model Train Karo
```
python run.py train
```

### Step 3 — Live Predict Karo
```
python run.py predict
```

---

## Age Groups
| Label | Group        | Age Range |
|-------|--------------|-----------|
| 0     | Child        | 0–12      |
| 1     | Teen         | 13–19     |
| 2     | Young Adult  | 20–35     |
| 3     | Middle Age   | 36–55     |
| 4     | Senior       | 56+       |

## Project Structure
```
age_recognition/
├── run.py
├── dataset/
│   ├── child/
│   ├── teen/
│   ├── young_adult/
│   ├── middle_age/
│   └── senior/
└── model/
    └── age_model.yml
```

## Tips
- Minimum **50-100 images per age group** for better accuracy
- Sabhi age groups ki images lo — jitne zyada groups, utni zyada images chahiye
- Sirf jo age groups train kiye hain unhi ke liye predict karega
