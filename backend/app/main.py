"""
SmartPrice — FastAPI Backend Application
=========================================
Main entrypoint configuring CORS, application lifespan, and endpoint routers.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .predictor import engine
from .routes import health, predict, models, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models and preprocessor pipeline
    print("Starting up SmartPrice API Engine...")
    engine.load()
    print("Models loaded and ready for predictions.")
    yield
    # Shutdown
    print("Shutting down SmartPrice API Engine.")


app = FastAPI(
    title="SmartPrice API",
    description="Machine Learning API for Smartphone Price Category Prediction & Multi-Model Benchmarking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Comma-separated browser origins. Configure CORS_ORIGINS in Render after the
# frontend static-site URL is known.
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers under /api
app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "Welcome to SmartPrice ML Prediction API",
        "docs": "/docs",
        "health": "/api/health"
    }
