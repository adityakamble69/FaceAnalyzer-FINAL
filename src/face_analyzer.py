"""
face_analyzer.py
================
Local .h5 models se Age, Gender, Emotion detect karta hai.

Folder structure:
    face_analyzer/
    ├── models/
    │   ├── age_model.h5
    │   ├── gender_model.h5
    │   └── emotion_model.h5
    ├── results/           ← results yahan save honge
    └── src/
        ├── face_analyzer.py   ← yahi run karo
        ├── local_models.py
        └── utils.py

Install karo:
    pip install deepface tf-keras opencv-python pillow numpy

Run karo:
    python src/face_analyzer.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image, ImageTk
from local_models import analyze_local

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── Theme ─────────────────────────────────────────────────────────────────────
BG      = "#0a0a0f"
SURFACE = "#13131a"
SUR2    = "#1c1c27"
BORDER  = "#2a2a3a"
ACCENT  = "#7c6ff7"
ACCENT2 = "#4ecdc4"
TEXT    = "#f0f0f5"
MUTED   = "#888899"
RED     = "#ff6b6b"

EMO_COLORS = {
    "angry":    "#ff6b6b",
    "disgust":  "#80b918",
    "fear":     "#c77dff",
    "happy":    "#4ecdc4",
    "sad":      "#6699ff",
    "surprise": "#ffd93d",
    "neutral":  "#888899",
}
EMO_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "sad": "😢", "surprise": "😮", "neutral": "😐"
}

# Results folder — project root ke andar
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bgr(img: Image.Image):
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_for_preview(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    tmp = img.copy()
    tmp.thumbnail((max_w, max_h), Image.LANCZOS)
    return tmp


# ── Main App ──────────────────────────────────────────────────────────────────
class FaceAnalyzerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Face Analyzer AI  ✦  Local Models")
        self.geometry("920x700")
        self.minsize(720, 560)
        self.configure(bg=BG)

        self._cap           = None
        self._cam_running   = False
        self._current_frame = None
        self._photo_img     = None
        self._after_id      = None
        self._last_result   = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=14, padx=20)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Face Analyzer AI",
                 font=("Helvetica", 20, "bold"), bg=BG, fg=TEXT).pack(side="left")
        tk.Label(hdr, text=" ✦ Local Models  •  No API  •  Offline",
                 font=("Helvetica", 11), bg=BG, fg=ACCENT).pack(side="left", padx=8)

        # Info bar
        info = tk.Frame(self, bg=SUR2, pady=8, padx=16)
        info.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(info,
                 text="✅  age_model.h5  •  gender_model.h5  •  emotion_model.h5  —  Sab local!",
                 font=("Helvetica", 10), bg=SUR2, fg=ACCENT2).pack(anchor="w")

        # Main grid
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────────────
        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self._preview = tk.Canvas(left, bg=SURFACE, bd=0,
                                  highlightthickness=1,
                                  highlightbackground=BORDER)
        self._preview.grid(row=0, column=0, sticky="nsew")
        self._draw_placeholder()

        # Buttons
        btns = tk.Frame(left, bg=BG)
        btns.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure((0, 1), weight=1)

        self._cam_btn = self._mk_btn(btns, "📷  Camera", ACCENT,
                                     self._toggle_camera, 0, 0)
        self._mk_btn(btns, "🖼   Upload", SUR2,
                     self._upload_photo, 0, 1)
        self._analyze_btn = self._mk_btn(btns, "🔍  Analyze Karo", ACCENT2,
                                         self._start_analyze, 1, 0,
                                         cs=2, fg="#000")
        self._analyze_btn.config(state="disabled")

        # Save button — analyze ke baad enable hoga
        self._save_btn = self._mk_btn(btns, "💾  Result Save Karo", SUR2,
                                      self._save_result, 2, 0, cs=2)
        self._save_btn.config(state="disabled")

        # Status
        self._status_var = tk.StringVar(value="Camera kholo ya photo upload karo")
        self._status_lbl = tk.Label(
            left, textvariable=self._status_var,
            bg=SURFACE, fg=MUTED, font=("Helvetica", 10),
            wraplength=260, anchor="w", padx=10, pady=8
        )
        self._status_lbl.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        # ── Right panel (scrollable results) ─────────────────────────────────
        ro = tk.Frame(main, bg=SURFACE, bd=0,
                      highlightthickness=1, highlightbackground=BORDER)
        ro.grid(row=0, column=1, sticky="nsew")
        ro.rowconfigure(0, weight=1)
        ro.columnconfigure(0, weight=1)

        self._rc = tk.Canvas(ro, bg=SURFACE, bd=0, highlightthickness=0)
        self._rc.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(ro, orient="vertical", command=self._rc.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._rc.configure(yscrollcommand=sb.set)

        self._rf = tk.Frame(self._rc, bg=SURFACE)
        self._rw = self._rc.create_window((0, 0), window=self._rf, anchor="nw")
        self._rf.bind("<Configure>", lambda e: self._rc.configure(
            scrollregion=self._rc.bbox("all")))
        self._rc.bind("<Configure>", lambda e: self._rc.itemconfig(
            self._rw, width=e.width))

        self._show_results_placeholder()

    def _mk_btn(self, parent, text, bg, cmd, r, c, cs=1, fg=TEXT):
        b = tk.Button(parent, text=text, bg=bg, fg=fg, relief="flat",
                      font=("Helvetica", 11, "bold"), cursor="hand2",
                      command=cmd, padx=10, pady=8,
                      activebackground=bg, activeforeground=fg)
        b.grid(row=r, column=c, columnspan=cs, sticky="ew", padx=3, pady=3)
        return b

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self._preview.delete("all")
        w = self._preview.winfo_width() or 280
        h = self._preview.winfo_height() or 320
        self._preview.create_text(w // 2, h // 2, text="📷",
                                  font=("Helvetica", 42), fill=MUTED)
        self._preview.create_text(w // 2, h // 2 + 54,
                                  text="Camera ya photo upload karo",
                                  font=("Helvetica", 11), fill=MUTED)

    def _show_results_placeholder(self):
        for w in self._rf.winfo_children():
            w.destroy()
        tk.Label(self._rf, text="Results yahaan aayenge ✦",
                 bg=SURFACE, fg=MUTED, font=("Helvetica", 13),
                 pady=60).pack(expand=True)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("Error", "pip install opencv-python")
            return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera", "Camera nahi mili!")
            return
        self._cam_running = True
        self._photo_img   = None
        self._cam_btn.config(text="✕  Band Karo")
        self._analyze_btn.config(state="normal")
        self._set_status("Camera chal rahi hai ✓", ACCENT2)
        self._update_camera()

    def _stop_camera(self):
        self._cam_running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self._cam_btn.config(text="📷  Camera")
        if not self._photo_img:
            self._analyze_btn.config(state="disabled")
            self._draw_placeholder()
        self._set_status("Camera band")

    def _update_camera(self):
        if not self._cam_running or not self._cap:
            return
        ret, frame = self._cap.read()
        if ret:
            self._current_frame = frame
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._show_on_canvas(pil)
        self._after_id = self.after(30, self._update_camera)

    # ── Upload ────────────────────────────────────────────────────────────────

    def _upload_photo(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All", "*.*")]
        )
        if not path:
            return
        self._stop_camera()
        img = Image.open(path)
        self._photo_img     = img
        self._current_frame = None
        self._show_on_canvas(img)
        self._analyze_btn.config(state="normal")
        self._set_status("Photo ready — analyze karo!", ACCENT2)

    # ── Preview ───────────────────────────────────────────────────────────────

    def _show_on_canvas(self, img: Image.Image):
        cw = self._preview.winfo_width()  or 280
        ch = self._preview.winfo_height() or 320
        preview = resize_for_preview(img, cw, ch)
        tkimg   = ImageTk.PhotoImage(preview)
        self._preview.delete("all")
        self._preview.create_image(cw // 2, ch // 2, anchor="center", image=tkimg)
        self._preview._tkimg = tkimg

    # ── Analyze ───────────────────────────────────────────────────────────────

    def _start_analyze(self):
        if self._photo_img is not None:
            frame = pil_to_bgr(self._photo_img)
        elif self._cam_running and self._current_frame is not None:
            frame = self._current_frame.copy()
        else:
            messagebox.showerror("Image", "Pehle camera kholo ya photo upload karo.")
            return

        self._analyze_btn.config(state="disabled", text="Analyzing...")
        self._save_btn.config(state="disabled")
        self._set_status("🔍 Models analyze kar rahe hain...", ACCENT)
        self._show_results_placeholder()

        threading.Thread(target=self._do_analyze, args=(frame,), daemon=True).start()

    def _do_analyze(self, frame):
        try:
            result = analyze_local(frame)
            self.after(0, self._show_result, result)
        except Exception as e:
            self.after(0, self._set_status, f"Error: {str(e)[:80]}", RED)
        finally:
            self.after(0, lambda: self._analyze_btn.config(
                state="normal", text="🔍  Analyze Karo"))

    # ── Show Result ───────────────────────────────────────────────────────────

    def _show_result(self, r: dict):
        if "error" in r:
            self._set_status(r["error"], RED)
            return

        self._last_result = r
        self._set_status("Analysis complete! ✦", ACCENT2)
        self._save_btn.config(state="normal")  # Save button enable

        for w in self._rf.winfo_children():
            w.destroy()

        pad = {"padx": 14, "pady": 6}

        # Stat cards
        crow = tk.Frame(self._rf, bg=SURFACE)
        crow.pack(fill="x", **pad)
        crow.columnconfigure((0, 1, 2), weight=1)

        emo   = r.get("dominant_emotion", "neutral")
        stats = [
            ("UMAR",    str(r.get("age", "—")), r.get("age_range", ""), ACCENT),
            ("GENDER",  r.get("gender", "—"),   "",                      ACCENT2),
            ("EMOTION", emo.capitalize(),        EMO_EMOJI.get(emo, ""), "#4ecdc4"),
        ]
        for i, (lbl, val, sub, col) in enumerate(stats):
            card = tk.Frame(crow, bg=SUR2, padx=10, pady=10)
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            tk.Label(card, text=lbl, bg=SUR2, fg=MUTED,
                     font=("Courier", 9)).pack()
            tk.Label(card, text=val, bg=SUR2, fg=col,
                     font=("Helvetica", 20, "bold")).pack()
            tk.Label(card, text=sub, bg=SUR2, fg=MUTED,
                     font=("Helvetica", 9)).pack()

        # Emotion bars
        tk.Label(self._rf, text="EMOTION BREAKDOWN", bg=SURFACE, fg=MUTED,
                 font=("Courier", 9), anchor="w"
                 ).pack(fill="x", padx=14, pady=(14, 4))

        for name, pct in sorted(r.get("emotions", {}).items(),
                                 key=lambda x: x[1], reverse=True):
            pct   = round(pct)
            color = EMO_COLORS.get(name, MUTED)
            row   = tk.Frame(self._rf, bg=SURFACE)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=name.capitalize(), bg=SURFACE, fg=MUTED,
                     font=("Helvetica", 10), width=10, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=SUR2, height=6)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 8))
            bar_bg.pack_propagate(False)
            tk.Frame(bar_bg, bg=color, height=6).place(
                relwidth=pct / 100, relheight=1)
            tk.Label(row, text=f"{pct}%", bg=SURFACE, fg=MUTED,
                     font=("Helvetica", 10), width=4).pack(side="right")

        self._rf.update_idletasks()

    # ── Save Result ───────────────────────────────────────────────────────────

    def _save_result(self):
        if not self._last_result:
            return

        # results/ folder banao agar nahi hai
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Timestamp se unique filename
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{ts}.json"
        filepath = os.path.join(RESULTS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._last_result, f, ensure_ascii=False, indent=2)

        self._set_status(f"✓ Saved: results/{filename}", ACCENT2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg, color=MUTED):
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    def _on_close(self):
        self._stop_camera()
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = FaceAnalyzerApp()
    app.mainloop()