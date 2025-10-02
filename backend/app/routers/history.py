from typing import List, Dict, Any, Optional
import datetime as dt

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..db import get_db
from .. import models

router = APIRouter(prefix="/api/v1/history", tags=["history"])

def _utcnow():
    return dt.datetime.utcnow()

@router.get("/{plate}")
def get_history_for_plate(
    plate: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Returns car metadata + sightings for the last `days` (default 30),
    plus a simple per-day summary.
    """
    plate = plate.upper().strip()
    car = db.get(models.Car, plate)
    if not car:
        raise HTTPException(status_code=404, detail=f"Plate '{plate}' not found")

    cutoff = _utcnow() - dt.timedelta(days=days)

    # fetch recent sightings (newest first)
    q = (
        db.query(models.ParkingHistory)
        .filter(models.ParkingHistory.plate == plate)
        .filter(models.ParkingHistory.timestamp >= cutoff)
        .order_by(desc(models.ParkingHistory.timestamp))
    )
    rows: List[models.ParkingHistory] = q.all()

    # serialize
    sightings: List[Dict[str, Any]] = []
    for r in rows:
        sightings.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "lot": r.lot,
            "image_url": r.image_url,
            "confidence": r.confidence,
            "bbox": r.bbox,
        })

    # per-day summary (Python-side; simple and clear)
    summary: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        day = r.timestamp.date().isoformat()
        if day not in summary:
            summary[day] = {"date": day, "counts": {"A": 0, "B": 0, "C": 0}, "last_seen": None}
        summary[day]["counts"][r.lot] = summary[day]["counts"].get(r.lot, 0) + 1
        # track last_seen for the day
        if not summary[day]["last_seen"] or r.timestamp > dt.datetime.fromisoformat(summary[day]["last_seen"]):
            summary[day]["last_seen"] = r.timestamp.isoformat()

    summary_list = sorted(summary.values(), key=lambda d: d["date"], reverse=True)

    return {
        "plate": plate,
        "car": {
            "plate": car.plate,
            "owner_name": car.owner_name,
            "owner_contact": car.owner_contact,
            "car_model": car.car_model,
            "notes": car.notes,
            "last_seen": car.last_seen.isoformat() if car.last_seen else None,
        },
        "range_days": days,
        "sightings_count": len(sightings),
        "sightings": sightings,
        "per_day": summary_list,
    }
