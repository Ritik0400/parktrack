from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/api/v1/plates", tags=["plates"])

@router.get("/{plate}", response_model=schemas.CarWithHistory)
def get_plate(plate: str, lot: Optional[str] = None, db: Session = Depends(get_db)):
    plate_norm = plate.strip().upper()
    car = db.query(models.Car).filter(models.Car.plate == plate_norm).first()
    q = db.query(models.ParkingHistory).filter(models.ParkingHistory.plate == plate_norm)
    if lot:
        q = q.filter(models.ParkingHistory.lot == lot.upper())
    history = q.order_by(models.ParkingHistory.timestamp.desc()).limit(200).all()
    return {"car": car, "history": history}
