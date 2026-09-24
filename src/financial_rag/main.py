"""FastAPI application factory for FinSight-RAG."""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="FinSight-RAG", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "").split(",") or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    # Routes will be mounted from api.routes
    from financial_rag.api.routes.auth import router as auth_router
    from financial_rag.api.routes.query import router as query_router
    app.include_router(auth_router, prefix="/auth", tags=["auth"])
    app.include_router(query_router, prefix="/query", tags=["query"])
    return app


app = create_app()
