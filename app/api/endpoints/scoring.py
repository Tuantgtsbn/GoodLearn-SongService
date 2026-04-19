"""
Scoring Endpoints
POST /api/v1/scoring/upload  - Upload audio và chấm điểm
POST /api/v1/scoring/score   - Chấm điểm từ file path (nội bộ)
"""
import os
import uuid
import aiofiles
import requests
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional

from app.core.config import settings
from app.core.logger import logger
from app.models.scoring import ScoringResult, ScoringError
from app.services.scoring_engine import scoring_engine
from app.services.beat_extractor import extract_beat

router = APIRouter()

# Định dạng file được chấp nhận
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".webm"}
MAX_FILE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024


def validate_audio_file(filename: str, file_size: Optional[int] = None) -> None:
    """Kiểm tra tính hợp lệ của file audio"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ: {ext}. Chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if file_size and file_size > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File quá lớn ({file_size / 1024 / 1024:.1f}MB). Tối đa {settings.MAX_FILE_SIZE_MB}MB",
        )


async def save_upload_file(file: UploadFile) -> str:
    """Lưu file upload vào thư mục tạm"""
    ext = Path(file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    return file_path


def cleanup_file(file_path: str) -> None:
    """Xóa file tạm sau khi xử lý"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Đã xóa file tạm: {file_path}")
    except Exception as e:
        logger.warning(f"Không xóa được file tạm {file_path}: {e}")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=ScoringResult,
    summary="Upload audio và chấm điểm giọng hát",
    description="""
Upload file audio giọng hát của bạn để nhận điểm số chi tiết.

**Các định dạng được hỗ trợ:** WAV, MP3, OGG, FLAC, M4A, AAC, WebM

**Tiêu chí chấm điểm:**
- 🎵 **Cao độ (40%)**: Độ chuẩn nốt nhạc, ổn định pitch
- 🥁 **Nhịp điệu (25%)**: Độ đều nhịp, cảm giác tempo
- 📏 **Ổn định (20%)**: Jitter, shimmer, hơi thở
- 🎭 **Biểu cảm (15%)**: Dynamics, cảm xúc

**Xếp loại:** S (90-100) | A (80-89) | B (70-79) | C (60-69) | D (50-59) | F (<50)
    """,
)
async def upload_and_score(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="File audio giọng hát"),
    song_title: Optional[str] = Form(None, description="Tên bài hát (tuỳ chọn)"),
    song_id: Optional[str] = Form(None, description="ID bài hát tham chiếu (tuỳ chọn)"),
    reference_audio_url: Optional[str] = Form(None, description="URL file audio gốc của bài hát"),
):
    """Upload và chấm điểm giọng hát"""
    logger.info(f"Nhận yêu cầu chấm điểm: {audio_file.filename}, song={song_title}, ref_url={reference_audio_url}")

    # Validate
    validate_audio_file(audio_file.filename)

    # Lưu file
    file_path = await save_upload_file(audio_file)
    # Chấm điểm
    try:
        result = scoring_engine.score(
            file_path,
            song_title=song_title,
            song_id=song_id,
            reference_audio_url=reference_audio_url,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi chấm điểm: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý audio: {str(e)}")
    finally:
        # Quan trọng: Dọn dẹp file audio tạm để tránh tốn dung lượng
        background_tasks.add_task(cleanup_file, file_path)

def process_async_scoring(
    file_path: str,
    webhook_url: str,
    song_title: Optional[str] = None,
    song_id: Optional[str] = None,
    reference_audio_url: Optional[str] = None,
):
    """Xử lý chấm điểm ngầm và gửi kết quả về webhook_url"""
    try:
        try:
            # Thực hiện chấm điểm (có thể mất thời gian)
            result = scoring_engine.score(
                file_path,
                song_title=song_title,
                song_id=song_id,
                reference_audio_url=reference_audio_url,
            )
            
            # Gửi webhook
            payload = result.dict()
            try:
                requests.post(webhook_url, json=payload, timeout=10)
                logger.info(f"Đã gửi kết quả chấm điểm tới webhook: {webhook_url}")
            except Exception as we_exc:
                logger.error(f"Lỗi khi gọi webhook {webhook_url}: {we_exc}")
                
        except Exception as e:
            logger.error(f"Lỗi chấm điểm (async): {e}", exc_info=True)
            # Gửi trạng thái FAILED về webhook
            try:
                requests.post(webhook_url, json={"status": "FAILED", "error": str(e)}, timeout=10)
            except Exception:
                pass
    finally:
        # Quan trọng: Xoá dọn dẹp file audio tạm sau khi hoàn tất để tránh tràn đĩa
        cleanup_file(file_path)


@router.post(
    "/async-upload",
    summary="Upload audio, chấm điểm bất đồng bộ",
    description="Nhận file, lập tức trả về status, đưa tác vụ vào queue và gọi lại bằng Webhook sau khi xong.",
)
async def async_upload_and_score(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="File audio giọng hát"),
    webhook_url: str = Form(..., description="URL để nhận webhook khi xử lý xong"),
    song_title: Optional[str] = Form(None, description="Tên bài hát (tuỳ chọn)"),
    song_id: Optional[str] = Form(None, description="ID bài hát tham chiếu (tuỳ chọn)"),
    reference_audio_url: Optional[str] = Form(None, description="URL file audio gốc của bài hát"),
):
    """Upload file và chấm điểm bằng background task"""
    logger.info(f"Nhận yêu cầu chấm điểm ASYNC: {audio_file.filename}, webhook={webhook_url}")

    # Validate
    validate_audio_file(audio_file.filename)

    # Lưu file
    file_path = await save_upload_file(audio_file)
    
    # Đưa vào background task
    background_tasks.add_task(
        process_async_scoring,
        file_path,
        webhook_url,
        song_title,
        song_id,
        reference_audio_url,
    )
    
    return {"status": "PROCESSING", "message": "File đang được xử lý", "file_path": file_path}


@router.get(
    "/supported-formats",
    summary="Lấy danh sách định dạng audio hỗ trợ",
)
async def get_supported_formats():
    return {
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "max_duration_seconds": settings.MAX_AUDIO_DURATION,
        "scoring_weights": {
            "pitch": settings.WEIGHT_PITCH,
            "rhythm": settings.WEIGHT_RHYTHM,
            "stability": settings.WEIGHT_STABILITY,
            "dynamics": settings.WEIGHT_DYNAMICS,
        },
    }


@router.delete(
    "/uploads/clear",
    summary="Xóa toàn bộ file trong thư mục uploads",
)
async def clear_uploads():
    """Xóa toàn bộ file trong thư mục uploads (không xóa thư mục con)."""
    deleted_files = 0
    failed_files = []

    try:
        if not os.path.exists(settings.UPLOAD_DIR):
            return {
                "message": "Thư mục uploads chưa tồn tại",
                "deleted_files": 0,
                "failed_files": [],
            }

        for name in os.listdir(settings.UPLOAD_DIR):
            file_path = os.path.join(settings.UPLOAD_DIR, name)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    deleted_files += 1
                except Exception as exc:
                    failed_files.append({"file": name, "error": str(exc)})

        return {
            "message": "Đã hoàn tất dọn uploads",
            "deleted_files": deleted_files,
            "failed_files": failed_files,
        }
    except Exception as exc:
        logger.error(f"Lỗi clear uploads: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Không thể dọn uploads: {exc}")


@router.post(
    "/extract-beat",
    summary="Tách nhạc beat (accompaniment)",
    description="Nhận file bài hát gốc, tiến hành xử lý để xoá giọng hát và trả về file beat trong thư mục uploads.",
)
async def extract_beat_api(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...)
):
    logger.info(f"Yêu cầu tách beat cho file: {audio_file.filename}")
    validate_audio_file(audio_file.filename)
    
    # 1. Lưu file gốc
    source_path = await save_upload_file(audio_file)
    
    # Lên kế hoạch xoá file gốc sau khi request chạy xong
    background_tasks.add_task(cleanup_file, source_path)
    
    # 2. Tạo đường dẫn file beat
    ext = Path(audio_file.filename).suffix.lower()
    beat_filename = f"beat_{uuid.uuid4()}{ext}"
    beat_path = os.path.join(settings.UPLOAD_DIR, beat_filename)
    
    # 3. Tiến hành tách
    result_path = extract_beat(source_path, beat_path)
    
    if not result_path:
        raise HTTPException(status_code=500, detail="Không thể tách beat. Cần cài đặt demucs hoặc spleeter.")
        
    return {
        "message": "Tách beat thành công",
        "beat_file_path": result_path,
        "original_file": audio_file.filename
    }
