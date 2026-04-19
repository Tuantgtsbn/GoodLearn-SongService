"""
API Router - Tập hợp tất cả endpoints
"""
from fastapi import APIRouter
from app.api.endpoints import scoring, songs

api_router = APIRouter()
api_router.include_router(scoring.router, prefix="/scoring", tags=["Scoring"])
api_router.include_router(songs.router, prefix="/songs", tags=["Songs"])
