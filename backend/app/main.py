from fastapi import FastAPI

app = FastAPI(title="ParkTrack API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "service": "parktrack-api"}
