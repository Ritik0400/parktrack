from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from ..db import get_db
from .. import models

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("")
def dashboard_view(db: Session = Depends(get_db)):
    """
    Real-time dashboard for frontend.
    Shows:
        - total detections today
        - lot-wise summary (A, B, C)
        - latest 50 sightings (joined with car metadata)
    """

    # --------------------------------------------------------
    # 1) LOT COUNTS (today)
    # --------------------------------------------------------
    lot_counts = {"A": 0, "B": 0, "C": 0}

    today_count_rows = (
        db.query(models.ParkingHistory.lot, func.count())
        .group_by(models.ParkingHistory.lot)
        .all()
    )

    for lot, count in today_count_rows:
        if lot in lot_counts:
            lot_counts[lot] = count

    total = sum(lot_counts.values())

    # --------------------------------------------------------
    # 2) LATEST 50 ENTRIES (join Car + ParkingHistory)
    # --------------------------------------------------------
    rows = (
        db.query(models.ParkingHistory, models.Car)
        .join(models.Car, models.Car.plate == models.ParkingHistory.plate)
        .order_by(desc(models.ParkingHistory.timestamp))
        .limit(50)
        .all()
    )

    entries = []
    for ph, car in rows:
        entries.append(
            {
                "plate": ph.plate,
                "last_seen": ph.timestamp.isoformat(),
                "lot": ph.lot,
                "owner_name": car.owner_name,
                "car_model": car.car_model,
            }
        )

    return {
        "total": total,
        "lots": lot_counts,
        "entries": entries,
    }
