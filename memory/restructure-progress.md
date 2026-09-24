---
name: restructuring-progress
description: Restructure to production-level package layout
metadata:
  type: project
---

Completed 2026-09-24:
- Copied all core modules (`auth`, `cache`, `embeddings`, `generation`/`llm`, `retrieval`/`vector_db`, `ingestion`, `processing`, `evaluation`) into `src/financial_rag/` subpackages.
- Updated all `from src.*` imports inside the new package to `from financial_rag.*` (auth, cache, embeddings, llm, vector_db, ingestion, processing, evaluation).
- Fixed `PROJECT_ROOT` depth (`parents[3]`) in `infrastructure/embeddings/embedder.py` and `application/evaluation/*.py`.
- Created `docker-compose.yml` (redis + api), `Makefile`, `main.py` app factory, `api/routes/auth.py` skeleton, and updated `Dockerfile` CMD.
- Old flat `src/*.py` kept intact (tests still point there); user confirmed this is acceptable for incremental restructuring.

Why: User requested production-level file structure (`src/financial_rag/...`) matching modular monolith pattern.
How to apply: Continue by splitting `src/api.py` into `routes/auth.py`, `routes/documents.py`, `routes/query.py`, and moving tests to reference new package.
