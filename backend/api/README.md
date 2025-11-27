# FastAPI Backend Service

Backend API service for processing dashcam videos with RT-DETR.

## Setup

1. **Install dependencies**:
```bash
cd backend/api
pip install -r requirements.txt
```

2. **Run server**:
```bash
python run_server.py
# or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. **API will be available at**:
```
http://localhost:8000
```

## API Endpoints

### `GET /`
Health check endpoint.

### `GET /health`
Health check with model status.

### `POST /upload`
Upload video for processing.

**Request:**
- `file`: Video file (multipart/form-data)
- `checkpoint_path` (optional): Path to model checkpoint
- `conf_threshold` (optional): Confidence threshold (default: 0.5)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Video uploaded successfully"
}
```

### `GET /status/{job_id}`
Get processing status.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 0.5,
  "message": "Processing video...",
  "output_file": "uuid_output.mp4"
}
```

### `GET /download/{job_id}`
Download processed video.

### `GET /jobs`
List all processing jobs.

## Usage Example

```bash
# Upload video
curl -X POST "http://localhost:8000/upload" \
  -F "file=@video.mp4" \
  -F "conf_threshold=0.5"

# Check status
curl "http://localhost:8000/status/{job_id}"

# Download result
curl "http://localhost:8000/download/{job_id}" -o output.mp4
```

