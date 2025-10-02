import os
import uuid
import json
import pathlib
import datetime as dt
import traceback
from typing import Optional, Any, Dict

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from .. import models

# ------------ Add repo root to sys.path (so "alpr" imports work) ------------
# .../backend/app/routers/upload.py  -> parents[3] == repo root (C:\parktrack)
import sys
HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
print(f"[upload] REPO_ROOT added to sys.path: {REPO_ROOT}")

USING_STUB = True
IMPORT_ERROR: Optional[str] = None

try:
    from alpr.pipeline import recognize_plates_from_image
    USING_STUB = False
    print("[upload] ALPR pipeline import: OK (tesseract_pipeline)")
except Exception as e:
    IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"[upload] ALPR pipeline import FAILED -> using stub. Reason: {IMPORT_ERROR}")
    traceback.print_exc()

    def recognize_plates_from_image(image_path: str):
        return [{"plate": "ABC1234", "confidence": 0.91, "bbox": [100, 120, 240, 60]}]

router = APIRouter(prefix="/api/v1", tags=["upload"])

STORAGE_DIR = os.getenv("STORAGE_DIR", "uploads")

def _ensure_storage_dir() -> pathlib.Path:
    backend_root = pathlib.Path(__file__).resolve().parents[2]  # .../backend
    base = backend_root / STORAGE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base

def _norm_bbox(b: Any) -> Optional[Any]:
    if b is None:
        return None
    if isinstance(b, (list, tuple, dict)):
        return b
    try:
        return json.loads(b)
    except Exception:
        return None

@router.post("/upload-image")
async def upload_image(
    lot: str = Form(..., min_length=1, max_length=1),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    lot = lot.upper()
    if lot not in ("A", "B", "C"):
        raise HTTPException(status_code=400, detail="lot must be one of A, B, C")

    # --- save file to disk ---
    storage_base = _ensure_storage_dir()
    ext = pathlib.Path(file.filename or "").suffix.lower() or ".jpg"
    uid = uuid.uuid4().hex
    saved_name = f"{uid}{ext}"
    saved_path = storage_base / saved_name

    content = await file.read()
    try:
        with open(saved_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save file: {e}")

    image_url = f"file://{saved_path}"

    # --- ingest log: pending ---
    request_id = uid
    ingest = models.IngestLog(
        request_id=request_id,
        image_url=str(image_url),
        status="pending",
        raw_response=None,
    )
    db.add(ingest)
    db.commit()
    db.refresh(ingest)

    # --- run ALPR (pipeline or stub) ---
    try:
        detections = recognize_plates_from_image(str(saved_path)) or []
    except Exception as e:
        ingest.status = "failed"
        ingest.raw_response = {"error": f"alpr_exception: {type(e).__name__}: {e}"}
        db.commit()
        raise HTTPException(status_code=500, detail="ALPR processing failed")

    # --- de-duplicate plates per request & skip NO_TEXT/empty ---
    # keep best (highest confidence) detection per plate string
    best_per_plate: Dict[str, Dict[str, Any]] = {}
    for det in detections:
        plate = str(det.get("plate", "")).strip().upper()
        if not plate or plate == "NO_TEXT":
            continue
        conf = float(det.get("confidence", 0.0) or 0.0)
        if (plate not in best_per_plate) or (conf > float(best_per_plate[plate].get("confidence", 0.0))):
            best_per_plate[plate] = {
                "plate": plate,
                "confidence": conf,
                "bbox": _norm_bbox(det.get("bbox")),
            }

    # --- write unique plates to DB ---
    written = []
    now = dt.datetime.utcnow()
    for plate, det in best_per_plate.items():
        # upsert car (avoid double-insert during same flush)
        car = db.get(models.Car, plate)
        if not car:
            car = models.Car(plate=plate)
            db.add(car)
        car.last_seen = now

        ph = models.ParkingHistory(
            plate=plate,
            lot=lot,
            image_url=str(image_url),
            confidence=float(det["confidence"]),
            bbox=det["bbox"],
        )
        db.add(ph)

        written.append(
            {"plate": plate, "lot": lot, "confidence": float(det["confidence"]), "bbox": det["bbox"]}
        )

    # --- finalize ingest log ---
    ingest.status = "success"
    ingest.raw_response = {
        "engine": "tesseract_pipeline" if not USING_STUB else "stub",
        "import_error": IMPORT_ERROR,
        "detections": detections,
    }
    db.commit()

    engine_used = "tesseract_pipeline" if not USING_STUB else "stub"
    return {
        "request_id": request_id,
        "saved": str(saved_path),
        "lot": lot,
        "engine": engine_used,
        "detections": written,  # only the DB-written, deduped plates
    }
