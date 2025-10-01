# ADR-0001: Use a monorepo for ParkTrack

## Decision
We will keep backend, ALPR, web, mobile, infra, db, and docs in a single repository.

## Status
Accepted

## Consequences
- Easier cross-cutting changes and shared CI.
- Larger repo size (mitigate with Git LFS for images/models).
