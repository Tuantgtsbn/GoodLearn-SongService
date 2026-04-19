import os
import shutil
import tempfile
import subprocess
import sys
import importlib
from pathlib import Path
from typing import Optional

from app.core.logger import logger
from app.core.config import settings

def extract_beat(source_path: str, output_filename: str) -> Optional[str]:
    """
    Tách lời ra khỏi nhạc, chỉ giữ lại phần beat (accompaniment).
    Lưu vào thư mục uploads với tên output_filename.
    Trả về đường dẫn file beat đã lưu hoặc None nếu thất bại.
    """
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Thử dùng spleeter (nếu có)
    has_spleeter = importlib.util.find_spec("spleeter.separator") is not None
    if has_spleeter:
        logger.info(f"Đang dùng Spleeter để tách beat cho {source_path}")
        with tempfile.TemporaryDirectory(prefix="spleeter_beat_") as tmp_dir:
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "spleeter",
                    "separate",
                    "-p",
                    "spleeter:2stems",
                    "-o",
                    tmp_dir,
                    source_path
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                stem_name = Path(source_path).stem
                beat_path = Path(tmp_dir) / stem_name / "accompaniment.wav"
                
                if beat_path.exists():
                    shutil.copy2(beat_path, output_filename)
                    return str(output_filename)
            except Exception as e:
                logger.error(f"Lỗi khi dùng Spleeter tách beat: {e}")
                
    # 2. Nếu không có spleeter, thử demucs
    has_demucs = importlib.util.find_spec("demucs") is not None
    if has_demucs:
        logger.info(f"Đang dùng Demucs để tách beat cho {source_path}")
        with tempfile.TemporaryDirectory(prefix="demucs_beat_") as tmp_dir:
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "demucs.separate",
                    "--two-stems=vocals",
                    "-o",
                    tmp_dir,
                    source_path
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # demucs sẽ tạo ra file out_dir / mdx_extra_q/stem_name / no_vocals.wav
                # Tùy version model mặc định (thường là htdemucs) nên dùng glob
                beat_file = None
                for path in Path(tmp_dir).glob("**/no_vocals.wav"):
                    beat_file = path
                    break
                    
                if beat_file and beat_file.exists():
                    shutil.copy2(beat_file, output_filename)
                    return str(output_filename)
            except Exception as e:
                logger.error(f"Lỗi khi dùng Demucs tách beat: {e}")
                
    logger.error("Cả Spleeter và Demucs đều không hoạt động hoặc không tách được beat.")
    return None
