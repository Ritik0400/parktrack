import os
import uuid
import pathlib
import datetime as dt
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from .. import models

# allow importing carid/* (repo root)
import sys
HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # C:\parktrack
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carid import detector, embedder, indexer  # type: ignore

router = APIRouter(prefix="/api/v1/reid", tags=["reid"])

# ---------- paths / storage ----------
STORAGE_DIR = os.getenv("STORAGE_DIR", "uploads")  # under backend/
def _backend_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]  # .../backend

def _ensure_storage_dir() -> pathlib.Path:
    base = _backend_root() / STORAGE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base

# where FAISS + meta live
INDEX_DIR = str(_backend_root() / "data" / "reid_index")
pathlib.Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

# ---------- global index ----------
_DIM = embedder.dim()  # 512 for ViT-B/32
_INDEX = indexer.CarIndex(root_dir=INDEX_DIR, dim=_DIM)

# ---------- helpers ----------
def _save_upload(file: UploadFile) -> pathlib.Path:
    base = _ensure_storage_dir()
    ext = pathlib.Path(file.filename or "").suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    p = base / f"{uid}{ext}"
    content = file.file.read()
    with open(p, "wb") as f:
        f.write(content)
    return p

def _upsert_car(db: Session, plate: str, owner_name: Optional[str], owner_contact: Optional[str], car_model: Optional[str], notes: Optional[str]):
    plate = plate.upper().strip()
    if not plate:
        return
    car = db.get(models.Car, plate)
    if not car:
        car = models.Car(plate=plate)
        db.add(car)
    # update metadata if provided
    if owner_name:
        car.owner_name = owner_name
    if owner_contact:
        car.owner_contact = owner_contact
    if car_model:
        car.car_model = car_model
    if notes:
        car.notes = notes
    car.last_seen = dt.datetime.utcnow()

# ---------- endpoints ----------
@router.post("/enroll-car")
async def enroll_car(
    plate: str = Form(...),
    owner_name: Optional[str] = Form(None),
    owner_contact: Optional[str] = Form(None),
    car_model: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Enroll ONE car into the re-id index.
    - Saves image
    - Detects vehicle (largest box)
    - Embeds crop with OpenCLIP
    - Adds vector to FAISS with meta (plate, model, etc.)
    - Upserts Car row in DB
    """
    # save upload
    path = _save_upload(file)
    image_url = f"file://{path}"

    # detect vehicles (largest first)
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=400, detail="bad image")
    boxes = detector.detect_vehicles(img)
    x, y, w, h, score, label = boxes[0]  # take largest candidate
    crop = img[y:y+h, x:x+w]

    # embed + add to index
    vec = embedder.embed_bgr_image(crop).reshape(1, -1)  # (1,512)
    metas = [{
        "plate": plate.upper(),
        "owner_name": owner_name,
        "owner_contact": owner_contact,
        "car_model": car_model,
        "notes": notes,
        "image_url": image_url,
        "bbox": [x, y, w, h],
    }]
    ids = _INDEX.add(vec, metas)

    # upsert car in DB
    _upsert_car(db, plate, owner_name, owner_contact, car_model, notes)
    db.commit()

    return {
        "enrolled_id": ids[0],
        "plate": plate.upper(),
        "detector_box": [x, y, w, h],
        "image_url": image_url,
        "meta": metas[0],
    }

import cv2
import numpy as np

@router.post("/identify-cars")
async def identify_cars(
    lot: str = Form(..., min_length=1, max_length=1),
    file: UploadFile = File(...),
    topk: int = Form(3),
    score_threshold: float = Form(0.30),  # cosine similarity threshold (0..1)
    db: Session = Depends(get_db),
):
    """
    Identify one or more vehicles in the image:
    - Detect vehicles
    - Embed each crop
    - Search FAISS for top-k matches
    - If best score >= threshold: treat as match, update history for top-1
    """
    lot = lot.upper()
    if lot not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="lot must be one of A, B, C")

    path = _save_upload(file)
    image_url = f"file://{path}"

    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=400, detail="bad image")

    boxes = detector.detect_vehicles(img)  # up to 20
    detections: List[Dict[str, Any]] = []

    # embed all crops
    vecs = []
    crops = []
    for (x, y, w, h, score, label) in boxes:
        crop = img[y:y+h, x:x+w]
        crops.append((x, y, w, h))
        v = embedder.embed_bgr_image(crop)
        vecs.append(v)
    if not vecs:
        return {"lot": lot, "count": 0, "detections": []}

    Q = np.vstack(vecs).astype("float32")  # (N,512)
    results = _INDEX.search(Q, k=int(topk))

    now = dt.datetime.utcnow()
    written = 0
    out = []
    for i, matches in enumerate(results):
        x, y, w, h = crops[i]
        top = matches[0] if matches else None
        match_plate = None
        match_score = None

        # if top match strong enough, treat as this car → write history + update last_seen
        if top and top.get("score", 0.0) >= float(score_threshold):
            md = top.get("meta", {})
            match_plate = (md.get("plate") or "").upper() or None
            match_score = top.get("score")

            if match_plate:
                car = db.get(models.Car, match_plate)
                if not car:
                    car = models.Car(plate=match_plate)
                    db.add(car)
                car.last_seen = now

                ph = models.ParkingHistory(
                    plate=match_plate,
                    lot=lot,
                    image_url=image_url,
                    confidence=float(match_score),
                    bbox=[x, y, w, h],
                )
                db.add(ph)
                written += 1

        out.append({
            "bbox": [x, y, w, h],
            "top_matches": matches,  # includes id, score, meta
            "matched_plate": match_plate,
            "matched_score": match_score,
        })

    db.commit()

    return {
        "lot": lot,
        "image_url": image_url,
        "count": len(boxes),
        "detections": out,
        "written_history": written,
    }
