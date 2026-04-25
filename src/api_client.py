"""
api_client.py
=============
Claude Vision API se face analysis karta hai.
"""

import base64
import json
import re
from io import BytesIO
from PIL import Image
import anthropic

SYSTEM_PROMPT = """You are a face analysis expert. Analyze the face in the image and return ONLY a JSON object.
No markdown, no backticks, no explanation — just raw JSON.

{
  "age": <number>,
  "age_range": "<e.g. 22-28>",
  "gender": "<Male/Female/Androgynous>",
  "gender_confidence": "<High/Medium/Low>",
  "dominant_emotion": "<happy/sad/angry/surprised/fearful/disgusted/neutral/excited/confused/calm>",
  "emotions": {
    "happy": <0-100>,
    "sad": <0-100>,
    "angry": <0-100>,
    "surprised": <0-100>,
    "neutral": <0-100>,
    "fearful": <0-100>
  },
  "notes": "<1 interesting observation in Hindi or English>"
}
If no face detected: {"error": "Koi chehra image mein nahi mila"}"""


def pil_to_base64(img: Image.Image) -> str:
    """PIL image ko base64 string mein convert karo."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def analyze_face(api_key: str, image_b64: str) -> dict:
    """
    Claude API ko call karo aur face analysis result lo.

    Args:
        api_key: Anthropic API key
        image_b64: Base64 encoded JPEG image

    Returns:
        dict with age, gender, emotion data
    """
    client = anthropic.Anthropic(api_key=api_key)

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": "Is image ka face analyze karo. Sirf JSON return karo."
                }
            ]
        }]
    )

    raw = "".join(b.text for b in msg.content if hasattr(b, "text"))
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return json.loads(match.group())
        raise ValueError("JSON parse nahi hua:\n" + raw[:300])
