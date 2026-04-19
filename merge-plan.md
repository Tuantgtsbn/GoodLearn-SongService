# Kế hoạch hợp nhất Vocal Scoring vào VieNeu-TTS

## Mục tiêu
Hợp nhất `vocal-scoring-backend` vào `VieNeu-TTS` thành một service chạy chung, có thể khởi động ổn định và gọi được các API chấm điểm giọng hát.

## Phạm vi công việc

### 1. Khảo sát source đích
- Kiểm tra cấu trúc FastAPI hiện tại của `VieNeu-TTS`.
- Xác định file entrypoint, router gốc, config, logging và cơ chế load `.env`.
- Xác định có đang dùng database, storage và background task nào sẵn không.

### 2. Hợp nhất cấu trúc backend
- Gộp `main.py`, `app/api`, `app/core`, `app/db`, `app/models`, `app/services`, `app/utils` vào source đích.
- Nối router scoring và songs vào router gốc của service.
- Đảm bảo không đè lên module đang có tên trùng trong `VieNeu-TTS`.

### 3. Gộp dependency
- Merge `requirements.txt` của vocal scoring vào dependency hiện có của `VieNeu-TTS`.
- Rà soát xung đột version giữa FastAPI, Pydantic, SQLAlchemy, NumPy, Librosa, SoundFile, MinIO, Demucs.
- Bổ sung dependency hệ thống nếu cần: `ffmpeg`, `libsndfile`.

### 4. Cấu hình môi trường
- Thêm các biến `.env` cần cho audio scoring, MinIO, database và CORS.
- Chuẩn hoá giá trị mặc định cho môi trường local và production.
- Kiểm tra đường dẫn `uploads/` và `data/reference_songs/` có được tạo tự động.

### 5. Database và schema
- Mang theo model `Song` và `SongScore` nếu service đích chưa có.
- Tạo migration hoặc cơ chế init schema phù hợp với DB hiện tại.
- Kiểm tra các bảng liên quan có bị trùng hoặc lệch kiểu dữ liệu.

### 6. Storage và file xử lý audio
- Kiểm tra MinIO/S3 bucket có sẵn hay cần tạo khi khởi động.
- Xác nhận luồng upload file tạm, dọn dẹp file, và xử lý async webhook.
- Kiểm tra quyền ghi vào thư mục `uploads/`.

### 7. Kiểm thử
- Chạy import check để phát hiện module trùng hoặc thiếu dependency.
- Test các route chính: `/health`, `/api/v1/scoring/upload`, `/api/v1/scoring/async-upload`, `/api/v1/songs`.
- Kiểm tra upload file thật với ít nhất 1 file WAV và 1 file MP3.

### 8. Docker và vận hành
- Đồng bộ `Dockerfile` và `docker-compose.yml` nếu `VieNeu-TTS` đang dùng container.
- Đảm bảo container có đủ package hệ thống cho audio processing.
- Kiểm tra log, healthcheck và restart policy.

## Thứ tự đề xuất
1. Khảo sát source đích.
2. Gộp dependency và config.
3. Gộp router, services, models.
4. Thêm migration/schema.
5. Chạy test cục bộ.
6. Sửa Docker và triển khai thử.
7. Kiểm tra lại API end-to-end.

## Tiêu chí hoàn thành
- Service khởi động không lỗi import.
- API scoring trả về kết quả hợp lệ.
- API songs đọc/ghi DB bình thường.
- Upload file hoạt động và file tạm được dọn sạch.
- Không xung đột với các tính năng hiện có của `VieNeu-TTS`.

## Rủi ro cần chú ý
- Xung đột dependency NumPy/Librosa/SciPy với source đích.
- Thiếu package hệ thống cho xử lý âm thanh.
- Trùng tên module hoặc router với code hiện có.
- Chưa có migration/schema tương ứng cho `Song` và `SongScore`.
- MinIO hoặc database chưa được cấu hình đúng trong môi trường chạy chung.
