"""
Vocal Scoring Backend - Main Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.router import api_router
from app.core.config import settings
from app.core.logger import logger
from app.db.session import Base, engine
from app.services.storage_service import storage_service
from app.db import models  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🎵 Vocal Scoring API đang khởi động...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    # Base.metadata.create_all(bind=engine)  # Bỏ việc tự động tạo table vì các table đã được migrate bên prisma
    try:
        storage_service.ensure_bucket()
    except Exception as exc:
        logger.warning(f"MinIO is not ready at startup: {exc}")
    yield
    logger.info("🛑 Vocal Scoring API đang dừng...")


app = FastAPI(
    title="Vocal Scoring API",
    description="Backend chấm điểm giọng hát thông minh",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "message": "🎵 Vocal Scoring API đang hoạt động",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
