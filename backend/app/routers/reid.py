import os
import uuid
import pathlib
import datetime as dt
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from .. import models

# -------------------------------------------------------------
# Allow import of carid/*
# -------------------------------------------------------------
import sys
HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]   # C:\parktrack
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import carid.detector as detector
import carid.embedder as embedder
import carid.indexer as indexer
import carid.color_classifier as color_classifier  # CLIP-based color classifier

router = APIRouter(prefix="/api/v1/reid", tags=["reid"])

# -------------------------------------------------------------
# Storage paths
# -------------------------------------------------------------
STORAGE_DIR = os.getenv("STORAGE_DIR", "uploads")


def _backend_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _ensure_storage_dir() -> pathlib.Path:
    base = _backend_root() / STORAGE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


# -------------------------------------------------------------
# FAISS INDEX
# -------------------------------------------------------------
INDEX_DIR = str(_backend_root() / "data" / "reid_index")
pathlib.Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

_DIM = embedder.dim()
_INDEX = indexer.CarIndex(root_dir=INDEX_DIR, dim=_DIM)

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _save_upload(file: UploadFile) -> pathlib.Path:
    base = _ensure_storage_dir()
    ext = pathlib.Path(file.filename or "").suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    p = base / f"{uid}{ext}"
    with open(p, "wb") as f:
        f.write(file.file.read())
    return p


def _position_label(x: int, w: int, img_width: int) -> str:
    """Compute left / center / right label."""
    cx = x + w / 2
    third = img_width / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "center"
    else:
        return "right"


# -------------------------------------------------------------
# ENROLL CAR  — with BLACKLIST support
# -------------------------------------------------------------
@router.post("/enroll-car")
async def enroll_car(
    plate: str = Form(...),
    owner_name: Optional[str] = Form(None),
    owner_contact: Optional[str] = Form(None),
    car_model: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    is_blacklisted: bool = Form(False),     # NEW FIELD
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Enroll a car into the re-ID index.
    - Detect largest car
    - Compute embedding
    - Add to FAISS
    - Save metadata + blacklist flag to DB
    """

    # Save file
    path = _save_upload(file)
    image_url = f"file://{path}"

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, "bad image")

    # Detect car
    boxes = detector.detect_vehicles(img)
    if not boxes:
        raise HTTPException(400, "No vehicle detected")

    # Use the largest vehicle for embedding
    x, y, w, h, det_score, det_label = boxes[0]
    crop = img[y:y+h, x:x+w]

    # Compute embedding
    vec = embedder.embed_bgr_image(crop).reshape(1, -1)

    # Metadata for FAISS
    meta = {
        "plate": plate.upper(),
        "owner_name": owner_name,
        "owner_contact": owner_contact,
        "car_model": car_model,
        "notes": notes,
        "is_blacklisted": is_blacklisted,     # NEW
        "image_url": image_url,
        "bbox": [x, y, w, h],
    }

    # Insert into FAISS index
    _INDEX.add(vec, [meta])

    # Upsert car in DB
    plate_up = plate.upper().strip()
    car = db.get(models.Car, plate_up)
    if not car:
        car = models.Car(plate=plate_up)
        db.add(car)

    # Update fields
    car.owner_name = owner_name
    car.owner_contact = owner_contact
    car.car_model = car_model
    car.notes = notes
    car.is_blacklisted = is_blacklisted    # NEW
    car.last_seen = dt.datetime.utcnow()

    db.commit()

    return {
        "status": "enrolled",
        "plate": plate_up,
        "image_url": image_url,
        "detector_box": [x, y, w, h],
        "meta": meta,
    }


# -------------------------------------------------------------
# IDENTIFY CARS — returns is_blacklisted flag
# -------------------------------------------------------------
@router.post("/identify-cars")
async def identify_cars(
    lot: str = Form(...),
    file: UploadFile = File(...),
    topk: int = Form(3),
    score_threshold: float = Form(0.95),
    db: Session = Depends(get_db),
):
    """
    Identify all cars in an image.
    Returns:
        - bbox
        - color + confidence
        - left/center/right position
        - matched_plate (if ≥95%)
        - matched_score (percentage)
        - is_blacklisted (NEW)
    """
    lot = lot.upper()
    if lot not in ("A", "B", "C"):
        raise HTTPException(400, "lot must be A/B/C")

    # Save image
    path = _save_upload(file)
    image_url = f"file://{path}"

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, "bad image")

    H, W = img.shape[:2]

    # Detect cars
    boxes = detector.detect_vehicles(img)
    if not boxes:
        return {"lot": lot, "count": 0, "detections": []}

    detections = []
    crops = []
    vecs = []

    # Sort cars left→right
    boxes.sort(key=lambda b: b[0])

    # Process each car
    for (x, y, w, h, det_score, det_label) in boxes:
        crop = img[y:y+h, x:x+w]
        crops.append((x, y, w, h))

        # Color classification
        color_label, color_conf = color_classifier.predict_car_color(crop)

        # Embedding for re-ID
        vec = embedder.embed_bgr_image(crop)
        vecs.append(vec)

        pos = _position_label(x, w, W)

        detections.append({
            "bbox": [x, y, w, h],
            "color": color_label,
            "color_confidence": round(float(color_conf), 3),
            "position": pos,
            "matched_plate": None,
            "matched_score": None,
            "is_blacklisted": False,   # default
        })

    # Run FAISS search
    Q = np.vstack(vecs).astype("float32")
    faiss_results = _INDEX.search(Q, k=int(topk))

    now = dt.datetime.utcnow()
    written = 0

    # Match results
    for i, matches in enumerate(faiss_results):
        det = detections[i]
        x, y, w, h = crops[i]

        top = matches[0] if matches else None
        raw_score = float(top.get("score", 0.0)) if top else 0.0

        # Passed accuracy gating
        if raw_score >= float(score_threshold):
            meta = top.get("meta", {})
            plate = (meta.get("plate") or "").upper() or None

            if plate:
                best_score_pct = round(raw_score * 100.0, 3)

                # Update DB
                car = db.get(models.Car, plate)
                if not car:
                    car = models.Car(plate=plate)
                    db.add(car)

                car.last_seen = now

                # Add parking history
                ph = models.ParkingHistory(
                    plate=plate,
                    lot=lot,
                    image_url=image_url,
                    confidence=raw_score,
                    bbox=[x, y, w, h],
                )
                db.add(ph)
                written += 1

                det["matched_plate"] = plate
                det["matched_score"] = best_score_pct
                det["is_blacklisted"] = bool(car.is_blacklisted)   # NEW

        det["top_matches"] = matches

    db.commit()

    return {
        "lot": lot,
        "image_url": image_url,
        "count": len(detections),
        "detections": detections,
        "written_history": written,
    }
