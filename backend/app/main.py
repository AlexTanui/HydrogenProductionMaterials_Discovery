from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import benchmarks, molecules, predictions
from app.core.config import settings
from app.models.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)
app.include_router(molecules.router)
app.include_router(benchmarks.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
