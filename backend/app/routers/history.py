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

# ================================================================
# GET HISTORY FOR ONE PLATE
# ================================================================
@router.get("/{plate}")
def get_history_for_plate(
    plate: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    plate = plate.upper().strip()
    car = db.get(models.Car, plate)
    if not car:
        raise HTTPException(status_code=404, detail=f"Plate '{plate}' not found")

    cutoff = _utcnow() - dt.timedelta(days=days)

    q = (
        db.query(models.ParkingHistory)
        .filter(models.ParkingHistory.plate == plate)
        .filter(models.ParkingHistory.timestamp >= cutoff)
        .order_by(desc(models.ParkingHistory.timestamp))
    )
    rows: List[models.ParkingHistory] = q.all()

    sightings = []
    for r in rows:
        sightings.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "lot": r.lot,
            "image_url": r.image_url,
            "confidence": r.confidence,
            "bbox": r.bbox,
        })

    summary: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        day = r.timestamp.date().isoformat()
        if day not in summary:
            summary[day] = {
                "date": day,
                "counts": {"A": 0, "B": 0, "C": 0},
                "last_seen": None
            }
        summary[day]["counts"][r.lot] += 1
        if not summary[day]["last_seen"] or r.timestamp.isoformat() > summary[day]["last_seen"]:
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


# ================================================================
# REAL-TIME DASHBOARD — Recent activity for ALL plates
# ================================================================
@router.get("/dashboard")
def dashboard(
    minutes: int = Query(30, ge=1, le=1440),
    db: Session = Depends(get_db)
):
    """
    Returns all sightings in the past `minutes`,
    newest first, with joined car metadata.
    """
    cutoff = _utcnow() - dt.timedelta(minutes=minutes)

    rows = (
        db.query(models.ParkingHistory)
        .filter(models.ParkingHistory.timestamp >= cutoff)
        .order_by(desc(models.ParkingHistory.timestamp))
        .limit(100)
        .all()
    )

    out = []
    for r in rows:
        car = db.get(models.Car, r.plate)
        out.append({
            "plate": r.plate,
            "lot": r.lot,
            "timestamp": r.timestamp.isoformat(),
            "confidence": float(r.confidence),
            "bbox": r.bbox,
            "image_url": r.image_url,
            "owner": car.owner_name if car else None,
            "model": car.car_model if car else None,
        })

    return out
