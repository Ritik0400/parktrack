from sqlalchemy import Column, String, Text, TIMESTAMP, BigInteger, Float, CHAR, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from .db import Base

class Car(Base):
    __tablename__ = "car"
    plate = Column(String(16), primary_key=True)
    owner_name = Column(String(255))
    owner_contact = Column(String(255))
    car_model = Column(String(128))
    notes = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    last_seen = Column(TIMESTAMP)

class ParkingHistory(Base):
    __tablename__ = "parking_history"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    plate = Column(String(16), ForeignKey("car.plate", ondelete="CASCADE"))
    lot = Column(CHAR(1))
    timestamp = Column(TIMESTAMP, server_default=func.now())
    image_url = Column(Text)
    bbox = Column(JSONB)
    confidence = Column(Float)

class IngestLog(Base):
    __tablename__ = "ingest_log"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(64))
    image_url = Column(Text)
    processed_at = Column(TIMESTAMP, server_default=func.now())
    processing_time_ms = Column(BigInteger)
    status = Column(String(32))
    raw_response = Column(JSONB)
