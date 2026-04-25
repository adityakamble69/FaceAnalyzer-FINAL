"""
utils.py
========
Image conversion aur helper functions.
"""

import base64
import json
import os
from datetime import datetime
from io import BytesIO
from PIL import Image

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def pil_to_base64(img: Image.Image) -> str:
    """PIL Image → base64 JPEG string."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def cv2_to_pil(frame) -> Image.Image:
    """OpenCV BGR frame → PIL Image."""
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python install karo: pip install opencv-python")
    import numpy as np
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def resize_for_preview(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Image ko preview ke liye resize karo aspect ratio maintain karte hue."""
    img_copy = img.copy()
    img_copy.thumbnail((max_w, max_h), Image.LANCZOS)
    return img_copy


def save_result(result: dict, results_dir: str = "results") -> str:
    """
    Analysis result ko JSON file mein save karo.

    Returns:
        Saved file ka path
    """
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"result_{timestamp}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return filename


def load_all_results(results_dir: str = "results") -> list:
    """
    results/ folder se saare saved results load karo.

    Returns:
        List of result dicts
    """
    if not os.path.exists(results_dir):
        return []

    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["_filename"] = fname
                    results.append(data)
            except Exception:
                pass
    return results


def get_api_key() -> str:
    """
    .env file ya environment variable se API key lo.
    """
    # .env file check karo
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    # Environment variable check karo
    return os.environ.get("ANTHROPIC_API_KEY", "")


EMO_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "surprised": "😮",
    "fearful": "😨", "disgusted": "🤢", "neutral": "😐",
    "excited": "🤩", "confused": "😕", "calm": "😌"
}

EMO_COLORS = {
    "happy": "#4ecdc4", "sad": "#6699ff", "angry": "#ff6b6b",
    "surprised": "#ffd93d", "fearful": "#c77dff", "neutral": "#888899",
    "disgusted": "#80b918", "excited": "#ff9f43", "confused": "#f9ca24",
    "calm": "#6c5ce7"
}
