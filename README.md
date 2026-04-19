# 🎤 Vocal Scoring Backend

Backend Python chấm điểm giọng hát thông minh dùng **FastAPI** + **librosa**.

---

## 🏗️ Kiến trúc

```
vocal-scoring/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── app/
│   ├── api/
│   │   ├── router.py                # Tổng hợp routes
│   │   └── endpoints/
│   │       ├── scoring.py           # POST /scoring/upload
│   │       └── songs.py             # CRUD bài hát
│   │
│   ├── core/
│   │   ├── config.py                # Pydantic Settings
│   │   └── logger.py
│   │
│   ├── models/
│   │   └── scoring.py               # Request/Response schemas
│   │
│   ├── services/
│   │   ├── audio_preprocessor.py    # Load, normalize, VAD
│   │   ├── pitch_analyzer.py        # pyin F0, vibrato, jitter
│   │   ├── rhythm_analyzer.py       # Beat tracking, tempo
│   │   ├── stability_dynamics_analyzer.py
│   │   └── scoring_engine.py        # Pipeline tổng hợp + feedback
│   │
│   └── utils/
│       └── audio_utils.py           # Helper functions
│
├── tests/
│   └── test_scoring.py              # Test suite đầy đủ
│
└── data/
    └── reference_songs/             # Audio gốc (tuỳ chọn)
```

---

## 🎯 Tiêu chí chấm điểm

| Tiêu chí | Trọng số | Thuật toán |
|----------|----------|-----------|
| 🎵 **Cao độ (Pitch)** | 40% | pyin F0, cents deviation, stability |
| 🥁 **Nhịp điệu (Rhythm)** | 25% | Beat tracking, onset detection, tempo |
| 📏 **Ổn định (Stability)** | 20% | Jitter, shimmer, tremolo, breathiness |
| 🎭 **Biểu cảm (Dynamics)** | 15% | Dynamic range, spectral centroid variation |

Hệ thống hỗ trợ thêm (qua feature flags):
- Vocal separation trước khi phân tích (auto backend)
- Căn chỉnh thời gian bằng DTW khi so sánh với reference

### Xếp loại
| Grade | Điểm | Ý nghĩa |
|-------|------|---------|
| **S** | 90-100 | Xuất sắc |
| **A** | 80-89  | Giỏi |
| **B** | 70-79  | Khá |
| **C** | 60-69  | Trung bình |
| **D** | 50-59  | Yếu |
| **F** | 0-49   | Kém |

---

## 🚀 Khởi động nhanh

### 1. Cài đặt môi trường

```bash
# Clone hoặc copy project
cd vocal-scoring

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài dependencies
pip install -r requirements.txt

# (Tuỳ chọn) Cài backend tách vocal
# pip install demucs
# pip install spleeter

# Cài ffmpeg (cần cho MP3, M4A)
# Ubuntu/Debian:
sudo apt install ffmpeg libsndfile1
# macOS:
brew install ffmpeg
```

### 2. Chạy server

#### **Option A: Chạy trực tiếp trên Windows**

**Bước 1: Chuẩn bị môi trường**
```powershell
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# PowerShell:
venv\Scripts\Activate.ps1
# CMD (Command Prompt):
venv\Scripts\activate.bat

# Cài dependencies
pip install -r requirements.txt
```

**Bước 2: Cấu hình file .env**
```powershell
# Copy file .env.example thành .env
Copy-Item .env.example .env
# Hoặc sử dụng Command Prompt:
# copy .env.example .env
```

**Bước 3: Chạy server**
```powershell
# Chạy FastAPI development server (có auto-reload)
uvicorn main:app --reload --port 8000
```

Server sẽ chạy tại: **http://localhost:8000**
- Swagger UI: **http://localhost:8000/docs**
- ReDoc: **http://localhost:8000/redoc**

**Tắt server:** Nhấn `Ctrl + C` trong terminal

---

#### **Option B: Chạy với Docker**

**Bước 1: Chuẩn bị Docker**
```powershell
# Kiểm tra Docker đã cài chưa
docker --version
docker-compose --version

# Nếu chưa cài, tải Docker Desktop từ: https://www.docker.com/products/docker-desktop
```

**Bước 2: Build và run với Docker Compose**
```powershell
# Build image và start container
docker-compose up --build

# Hoặc chạy ở background (detached mode)
docker-compose up --build -d
```

**Bước 3: Truy cập server**
- API: **http://localhost:8000**
- Swagger UI: **http://localhost:8000/docs**

**Lệnh hữu ích với Docker:**
```powershell
# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Xóa containers và volumes
docker-compose down -v

# Chạy container đơn lẻ
docker-compose up -d vocal-scoring-api

# Check status
docker-compose ps

# Truy cập shell bên trong container
docker-compose exec vocal-scoring-api bash
```

---

**So sánh hai phương pháp:**

| Tiêu chí | Windows Native | Docker |
|----------|---|---|
| **Cài đặt** | Nhanh, cục bộ | Hơi phức tạp nhưng chuẩn hóa |
| **Performance** | ⚡ Nhanh hơn | Bình thường (thêm overhead) |
| **Dev Experience** | ✅ Tốt (auto-reload) | ✅ Tốt (isolated) |
| **Production** | Khó quản lý | ✅ Tốt (containers) |
| **Dependency Issues** | Có thể gặp vấn đề | ✅ Biến mất |
| **Hợp tác nhóm** | Có thể khác nhau | ✅ Nhất quán |

**Khuyến nghị:** 
- **Phát triển:** Sử dụng Windows Native vì nhanh và dễ debug
- **Production/CI-CD:** Sử dụng Docker để đảm bảo consistency

### 3. Test pipeline

```bash
python tests/test_scoring.py
```

---

## 📡 API Endpoints

### `POST /api/v1/scoring/upload`
Upload audio và nhận điểm số.

**Form Data:**
- `audio_file` *(file, required)*: File audio giọng hát
- `song_title` *(string, optional)*: Tên bài hát
- `song_id` *(string, optional)*: ID bài hát tham chiếu đã tạo ở `/api/v1/songs/`

**Định dạng hỗ trợ:** WAV, MP3, OGG, FLAC, M4A, AAC, WebM

**Ví dụ curl:**
```bash
curl -X POST http://localhost:8000/api/v1/scoring/upload \
  -F "audio_file=@my_voice.wav" \
  -F "song_title=Bài hát của tôi"
```

**Response mẫu:**
```json
{
  "song_title": "Bài hát của tôi",
  "audio_duration_seconds": 45.3,
  "sample_rate": 22050,
  "total_score": 78.5,
  "grade": "B",
  "pitch": {
    "score": 82.1,
    "average_pitch_hz": 261.5,
    "average_pitch_note": "C4",
    "pitch_accuracy_percent": 82.1,
    "out_of_tune_segments": 2,
    "pitch_stability": 0.87,
    "vocal_range": "Tenor",
    "voiced_ratio": 0.82
  },
  "rhythm": {
    "score": 75.3,
    "estimated_tempo_bpm": 95.0,
    "beat_consistency": 0.81,
    "onset_regularity": 0.76,
    "rhythm_deviation_ms": 45.2
  },
  "stability": {
    "score": 70.8,
    "vibrato_rate_hz": 5.5,
    "vibrato_extent_semitones": 0.8,
    "tremolo_detected": false,
    "breathiness_score": 0.15,
    "jitter_percent": 0.8,
    "shimmer_percent": 2.1
  },
  "dynamics": {
    "score": 80.2,
    "dynamic_range_db": 22.5,
    "loudness_variation": 0.45,
    "emotional_expressiveness": 0.72,
    "rms_energy_mean": 0.0023
  },
  "segments": [
    { "start_time": 0.0, "end_time": 5.0, "pitch_score": 85.0, "overall_score": 85.0, "feedback": "✅ Tuyệt vời" },
    { "start_time": 5.0, "end_time": 10.0, "pitch_score": 60.0, "overall_score": 60.0, "feedback": "⚠️ Cần cải thiện" }
  ],
  "feedback": {
    "strengths": ["🎵 Cao độ rất chuẩn (82/100)", "🥁 Giữ nhịp tốt"],
    "improvements": ["🎭 Thêm biểu cảm vào các đoạn slow"],
    "tips": ["💡 Tập vibrato để giọng thêm màu sắc"],
    "overall_comment": "⭐ Giỏi! 78.5 điểm - Giọng hát rất tốt."
  },
  "processing_time_ms": 1250.0
}
```

### `GET /api/v1/scoring/supported-formats`
Lấy danh sách định dạng và config hệ thống.

### `GET /api/v1/songs/`
Lấy danh sách bài hát.

### `POST /api/v1/songs/`
Thêm bài hát (kèm audio gốc để chấm điểm so sánh).

### `GET /api/v1/songs/{id}`
Chi tiết bài hát.

### `DELETE /api/v1/songs/{id}`
Xóa bài hát.

### Swagger UI
Truy cập: `http://localhost:8000/docs`

---

## ⚙️ Cấu hình

Chỉnh sửa `.env` để thay đổi trọng số chấm điểm:

```env
WEIGHT_PITCH=0.40      # Cao độ
WEIGHT_RHYTHM=0.25     # Nhịp điệu
WEIGHT_STABILITY=0.20  # Ổn định
WEIGHT_DYNAMICS=0.15   # Biểu cảm

# Feature flags
ENABLE_VOCAL_SEPARATION=false
ENABLE_DTW_ALIGNMENT=false

# Vocal separator backend
VOCAL_SEPARATOR_BACKEND=auto   # auto|demucs|spleeter|none
VOCAL_SEPARATOR_DEVICE=auto    # auto|cpu|cuda

# DTW alignment
DTW_MAX_FRAMES=3000
DTW_PENALTY=0.1
```

---

## 🔬 Thuật toán chính

| Module | Thuật toán |
|--------|-----------|
| Pitch F0 | **pyin** (probabilistic YIN, librosa) |
| Beat tracking | **Dynamic programming** beat tracker (librosa) |
| Onset detection | **Spectral flux** onset strength |
| Vibrato | **FFT** trên detrended MIDI values |
| Jitter/Shimmer | Biến động chu kỳ / biên độ frame-to-frame |
| Breathiness | **Spectral flatness** (HNR proxy) |
| Dynamics | **Spectral centroid** + RMS variation |

---

## 📦 Stack

- **FastAPI** 0.115 - Web framework
- **librosa** 0.10 - Audio analysis
- **numpy / scipy** - Signal processing
- **soundfile** - Audio I/O
- **pydantic** 2.x - Data validation
- **uvicorn** - ASGI server
