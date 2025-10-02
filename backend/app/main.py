from fastapi import FastAPI
from .routers import plates

app = FastAPI(title="ParkTrack API", version="0.2.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "parktrack-api"}

app.include_router(plates.router)
