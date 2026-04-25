# Face Analyzer AI

Age, Gender aur Emotion detect karta hai — Claude Vision API se.

## Folder Structure

```
face_analyzer/
├── src/
│   ├── face_analyzer.py   ← Main app (yahi run karo)
│   ├── api_client.py      ← Claude API calls
│   └── utils.py           ← Helper functions
├── assets/                ← Test images yahan rakhna
├── results/               ← Analysis results save honge
├── .env                   ← API key (secret!)
├── requirements.txt       ← Libraries
└── README.md
```

## Setup

### 1. Libraries install karo
```bash
pip install -r requirements.txt
```

### 2. API Key set karo
`.env` file mein apni key daalo:
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```
Key yahan se banao: https://console.anthropic.com/settings/keys

### 3. Run karo
```bash
python src/face_analyzer.py
```

## Features
- Live webcam se face analyze karo
- Photo upload karke analyze karo
- Age, Gender, Emotion detect hota hai
- Results JSON mein save kar sakte ho (`results/` folder mein)
- API key `.env` se auto-load hoti hai
