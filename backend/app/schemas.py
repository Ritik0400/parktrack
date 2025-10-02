from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime

class CarOut(BaseModel):
    plate: str
    owner_name: Optional[str] = None
    owner_contact: Optional[str] = None
    car_model: Optional[str] = None
    last_seen: Optional[datetime] = None
    class Config:
        from_attributes = True

class ParkingHistoryOut(BaseModel):
    id: int
    plate: str
    lot: str
    timestamp: datetime
    image_url: Optional[str] = None
    bbox: Optional[Any] = None
    confidence: Optional[float] = None
    class Config:
        from_attributes = True

class CarWithHistory(BaseModel):
    car: Optional[CarOut]
    history: List[ParkingHistoryOut] = []
