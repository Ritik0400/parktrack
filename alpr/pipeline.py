import os
import re
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pytesseract

# If uvicorn can't find tesseract.exe, uncomment:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# US-like plate heuristic: 5–8 chars, alphanumeric, must include a digit
PLATE_REGEX = re.compile(r"^[A-Z0-9]{5,8}$")

# ----------------------------- OCR helpers -----------------------------
def _ocr_plate(crop_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 15
    )
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, k, iterations=1)

    cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(th, config=cfg)
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    return text

def _confidence_heuristic(text: str, crop_wh: Tuple[int, int], img_wh: Tuple[int, int]) -> float:
    if not text:
        return 0.0
    W, H = img_wh
    w, h = crop_wh
    length_score = min(len(text) / 7.0, 1.0)  # 6–7 chars typical → ~1.0
    regex_bonus = 0.25 if PLATE_REGEX.match(text) and any(ch.isdigit() for ch in text) else 0.0
    size_bonus = min((w * h) / float(W * H) * 3.0, 0.25)
    conf = 0.5 * length_score + regex_bonus + size_bonus
    return float(max(0.05, min(conf, 0.98)))

# ----------------------- OpenCV heuristic fallback ----------------------
def _preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.Canny(blur, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def _candidate_boxes(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    closed = _preprocess(img)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    boxes: List[Tuple[int,int,int,int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < (W * H) * 0.004:
            continue
        ar = w / float(h) if h else 0
        if ar < 1.5 or ar > 7.0:
            continue
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.20)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    if not boxes:
        boxes = [(0, 0, W, H)]
    cx, cy = W / 2, H / 2
    boxes.sort(key=lambda b: (-(b[2]*b[3]), ((b[0]+b[2]/2-cx)**2+(b[1]+b[3]/2-cy)**2)**0.5))
    return boxes[:5]

def _detect_with_opencv(img: np.ndarray):
    boxes = _candidate_boxes(img)
    return [(x, y, w, h, 0.15) for (x, y, w, h) in boxes]

# ------------------------ YOLO (optional) support -----------------------
_yolo_model = None
def _try_load_yolo() -> Optional[object]:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    weights = os.getenv("YOLO_WEIGHTS", "").strip()
    if not weights:
        here = os.path.dirname(__file__)
        local_path = os.path.join(here, "models", "lp-yolov8n.pt")
        if os.path.isfile(local_path):
            weights = local_path

    if not weights or not os.path.isfile(weights):
        print("[alpr] YOLO weights not provided or file not found. Using OpenCV fallback.")
        _yolo_model = None
        return None

    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(weights)
        print(f"[alpr] YOLO loaded: {weights}")
        return _yolo_model
    except Exception as e:
        print(f"[alpr] Failed to load YOLO weights '{weights}': {e}")
        _yolo_model = None
        return None

def _detect_with_yolo(img: np.ndarray):
    model = _try_load_yolo()
    if model is None:
        return []
    H, W = img.shape[:2]
    results = model.predict(source=img, conf=0.25, verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(getattr(b.conf[0], "item", lambda: b.conf[0])())
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(W, int(x2)), min(H, int(y2))
            w, h = max(1, x2i - x1i), max(1, y2i - y1i)
            out.append((x1i, y1i, w, h, conf))
    return out

# ------------------------------ entry point -----------------------------
def recognize_plates_from_image(image_path: str) -> List[Dict]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    H, W = img.shape[:2]
    dets = _detect_with_yolo(img)
    if not dets:
        dets = _detect_with_opencv(img)

    candidates: List[Dict] = []
    for (x, y, w, h, det_conf) in dets:
        crop = img[y:y+h, x:x+w]
        text = _ocr_plate(crop) or "NO_TEXT"
        conf = 0.7 * _confidence_heuristic(text, (w, h), (W, H)) + 0.3 * float(det_conf)
        candidates.append({"plate": text, "confidence": round(conf, 2), "bbox": [x, y, w, h]})

    # sort by confidence
    candidates.sort(key=lambda d: d["confidence"], reverse=True)

    # keep only plate-like strings: 5–8 chars, includes at least one digit
    filtered = [
        c for c in candidates
        if PLATE_REGEX.match(c["plate"]) and any(ch.isdigit() for ch in c["plate"]) and c["confidence"] >= 0.55
    ]

    # dedupe exact same (plate, bbox)
    out: List[Dict] = []
    seen = set()
    for c in filtered:
        key = (c["plate"], tuple(c["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    return out[:5]
