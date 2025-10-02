import cv2
from typing import List, Tuple

# YOLO loaded lazily
_YOLO = None

# COCO labels we consider "vehicles"
VEHICLE_IDS = {2, 3, 5, 7}  # car=2, motorcycle=3, bus=5, truck=7

def _load():
    global _YOLO
    if _YOLO is not None:
        return _YOLO
    try:
        from ultralytics import YOLO
        _YOLO = YOLO("yolov8n.pt")  # downloads once; if it fails, we'll fallback
        print("[reid] YOLO COCO loaded")
    except Exception as e:
        print(f"[reid] YOLO COCO load failed: {e}")
        _YOLO = None
    return _YOLO

def detect_vehicles(bgr) -> List[Tuple[int,int,int,int,float,str]]:
    """
    Returns a list of (x, y, w, h, score, label).
    If nothing found or YOLO unavailable, returns one full-image box.
    """
    H, W = bgr.shape[:2]
    model = _load()
    boxes: List[Tuple[int,int,int,int,float,str]] = []

    if model is not None:
        results = model.predict(source=bgr, conf=0.25, verbose=False)
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
                if cls not in VEHICLE_IDS:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(W, int(x2)), min(H, int(y2))
                w, h = max(1, x2i - x1i), max(1, y2i - y1i)
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                label = names.get(cls, str(cls))
                boxes.append((x1i, y1i, w, h, conf, label))

    if not boxes:
        boxes = [(0, 0, W, H, 0.01, "full")]

    boxes.sort(key=lambda t: t[2] * t[3], reverse=True)  # biggest first
    return boxes[:20]
