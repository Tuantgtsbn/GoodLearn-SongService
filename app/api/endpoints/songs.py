"""
Songs Endpoints
Quản lý danh sách bài hát tham chiếu (reference songs)
"""
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, List
from sqlalchemy.orm import Session
import librosa
from pydantic import BaseModel, Field, ConfigDict

from app.core.logger import logger
from app.db.models import Song
from app.db.session import get_db
from app.services.storage_service import storage_service

router = APIRouter()


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class SongInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    title: str
    artists: Optional[List[str]] = None
    genre: Optional[str] = None
    duration_seconds: Optional[float] = None
    has_reference_audio: bool = False
    created_at: Optional[datetime] = None


class SongListResponse(BaseModel):
    total: int
    songs: List[SongInfo]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@router.get(
    "/",
    response_model=SongListResponse,
    summary="Lấy danh sách bài hát",
)
async def list_songs(db: Session = Depends(get_db)):
    songs_db = db.query(Song).order_by(Song.created_at.desc()).all()
    songs = [
        SongInfo(
            id=s.id,
            title=s.title,
            artists=s.artists,
            genre=s.genre,
            duration_seconds=s.duration_seconds,
            has_reference_audio=s.has_reference_audio,
            created_at=s.created_at,
        )
        for s in songs_db
    ]
    return SongListResponse(total=len(songs), songs=songs)


@router.get(
    "/{song_id}",
    response_model=SongInfo,
    summary="Lấy thông tin bài hát",
)
async def get_song(song_id: str, db: Session = Depends(get_db)):
    song = db.query(Song).filter(Song.id == song_id).first()
    if not song:
        raise HTTPException(status_code=404, detail="Bài hát không tồn tại")
    return SongInfo(
        id=song.id,
        title=song.title,
        artists=song.artists,
        genre=song.genre,
        duration_seconds=song.duration_seconds,
        has_reference_audio=song.has_reference_audio,
        created_at=song.created_at,
    )
