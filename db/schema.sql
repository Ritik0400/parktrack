-- Initial schema for ParkTrack

CREATE TABLE IF NOT EXISTS car (
  plate VARCHAR(16) PRIMARY KEY,
  owner_name VARCHAR(255),
  owner_contact VARCHAR(255),
  car_model VARCHAR(128),
  notes TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  last_seen TIMESTAMP
);

CREATE TABLE IF NOT EXISTS parking_history (
  id BIGSERIAL PRIMARY KEY,
  plate VARCHAR(16) REFERENCES car(plate) ON DELETE CASCADE,
  lot CHAR(1) CHECK (lot IN ('A','B','C')),
  timestamp TIMESTAMP DEFAULT NOW(),
  image_url TEXT,
  bbox JSONB,
  confidence REAL
);

CREATE INDEX IF NOT EXISTS idx_parking_history_plate_time
  ON parking_history (plate, timestamp DESC);

CREATE TABLE IF NOT EXISTS ingest_log (
  id BIGSERIAL PRIMARY KEY,
  request_id VARCHAR(64),
  image_url TEXT,
  processed_at TIMESTAMP DEFAULT NOW(),
  processing_time_ms INT,
  status VARCHAR(32),
  raw_response JSONB
);
