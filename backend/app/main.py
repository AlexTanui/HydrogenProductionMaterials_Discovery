from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import benchmarks, data, jobs, molecules, predictions
from app.core.config import settings
from app.models.db import init_db
from app.services.inference import warm_up
from app.services.jobs import recover_jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Load every phase's checkpoint now, not on whichever request hits it
    # first - so a missing/broken checkpoint shows up in the startup log
    # before a live demo, not mid-click.
    print("Warming up model checkpoints:", flush=True)
    for line in warm_up():
        print(line, flush=True)
    recovered = recover_jobs()
    if recovered:
        print(f"Recovered {len(recovered)} training job(s) from a previous run: {recovered}", flush=True)
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
app.include_router(jobs.router)
app.include_router(data.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
