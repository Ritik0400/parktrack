from fastapi import FastAPI
from .routers import plates
from .routers import upload
from .routers import reid
from .routers import history


app = FastAPI(title="ParkTrack API", version="0.2.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "parktrack-api"}

app.include_router(plates.router)
app.include_router(upload.router)
app.include_router(reid.router)
app.include_router(history.router)
