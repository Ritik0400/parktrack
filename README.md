# ParkTrack: Smart Parking Lot Monitoring (ALPR)

This repository contains a **monorepo** for a university project that detects license plates from parking-lot photos (mobile + web), recognizes plates, and stores/query parking history.

## Tech Stack (initial)
- **Backend API**: FastAPI (Python)
- **ALPR engine**: YOLO (detector) + OCR (EasyOCR / Tesseract)
- **Database**: PostgreSQL
- **Web**: React
- **Mobile**: React Native (placeholder to be scaffolded later)
- **Infra**: Docker Compose for DB & local dev

## Repo Structure
```
.
├── alpr/                # ALPR pipeline code (detector + OCR)
├── backend/             # FastAPI app
├── web/                 # React web app (dashboard)
├── mobile/              # React Native app (to be initialized)
├── db/                  # SQL schema, migrations, seed data
├── infra/               # Docker compose, deployment scripts
├── docs/                # Design docs, ADRs, diagrams
├── .gitignore
└── README.md
```

## Version Control Policy
- **Branching**: `main` is stable. Feature work in short-lived branches: `feat/*`, `fix/*`, `chore/*`.
- **Commits**: We only commit after **major features** or checkpoints. Use **Conventional Commits**:
  - `feat(api): add upload-image endpoint`
  - `feat(alpr): integrate yolov8 detector`
  - `chore(db): add initial schema`
- **Tags**: Semantic versioning tags for milestones: `v0.1.0`, `v0.2.0`, etc.
- **Pull Requests**: Even solo, open PRs for clarity and a paper trail (link issues to PRs).

## Getting Started (local, minimal)
1) Install:
   - Git, Python 3.10+, Node.js 18+, Docker Desktop, PostgreSQL (or use Docker), Tesseract (later).
2) Create and activate a Python venv in `backend/`:
   ```bash
   cd backend
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
3) Start DB (Docker):
   ```bash
   cd infra
   docker compose up -d
   ```
4) Apply schema:
   ```bash
   psql postgresql://postgres:postgres@localhost:5432/parktrack -f db/schema.sql
   ```

## Next Steps
- Implement `/api/v1/upload-image` + connect to ALPR stub.
- Scaffold React web app.
- Integrate EasyOCR/Tesseract and YOLO (alpr/).

## License
MIT (for academic use).
