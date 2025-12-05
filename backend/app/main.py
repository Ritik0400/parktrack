from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import plates
from .routers import upload
from .routers import reid
from .routers import history

from .routers import dashboard

app = FastAPI(
    title="ParkTrack API",
    version="0.2.0",
)

# ----------------------------------------------------
# CORS CONFIGURATION (Required for frontend to work)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods
    allow_headers=["*"],        # Allow all headers
)

# ----------------------------------------------------
# Health Check
# ----------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "parktrack-api"}

# ----------------------------------------------------
# Routers
# ----------------------------------------------------
app.include_router(plates.router)
app.include_router(upload.router)
app.include_router(reid.router)
app.include_router(history.router)
app.include_router(dashboard.router)
