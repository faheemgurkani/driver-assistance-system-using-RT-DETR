# FastAPI Backend Service

Backend API service for processing dashcam videos with RT-DETR and ADAS alerts.

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

## Features

- **Model**: RT-DETR with ResNet-101-VD backbone
- **ADAS Integration**: Blind spot detection and collision warning
- **Automatic Re-encoding**: H.264/AAC conversion for browser compatibility
- **Job Recovery**: Automatic recovery of completed jobs after server reloads
- **Prediction Logging**: Detailed JSON logs with ADAS alert information
- **HTTP Range Support**: Video streaming with partial content requests

## ADAS Modules

The backend uses modular ADAS components located in `backend/Alerts/modules/`:

- **Blind Spot Detection** (`blind_spot/blind_spot.py`): Detects vehicles in left/right blind spot zones
- **Collision Warning** (`collision/collision_warning.py`): Estimates frontal collision risk based on distance heuristics

These modules are automatically imported and used during video processing. If ADAS modules are unavailable, the system gracefully falls back to standard object detection without alerts.

## API Endpoints

### `GET /`
Health check endpoint.

### `GET /health`
Health check with model status.

### `POST /upload`
Upload video for processing.

**Request:**
- `file`: Video file (multipart/form-data)
- `pipeline_type` (optional): "original" or "saliency" (default: "original")
- `checkpoint_path` (optional): Path to model checkpoint (auto-detected if not provided)
- `conf_threshold` (optional): Confidence threshold (default: 0.5)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Video uploaded successfully"
}
```

**Checkpoint Auto-Detection:**
- For "saliency" pipeline: Searches `backend/output/d2city_saliency_enhanced_rtdetr_r101vd/` (preferred) or `d2city_saliency_enhanced_rtdetr_r50vd/`
- For "original" pipeline: Searches `backend/output/d2city_rtdetr_r101vd/` (preferred) or `d2city_rtdetr_r50vd/`
- Fallback: Uses any `.pth` file from `backend/checkpoints/`

### `GET /status/{job_id}`
Get processing status. Automatically recovers completed jobs from filesystem if not in memory.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 0.5,
  "message": "Processing video...",
  "output_file": "uuid_output.mp4",
  "log_file": "uuid_output_predictions.json"
}
```

**Status Messages:**
- "ADAS: Mapping blind-spot and collision risks per frame..." - ADAS alerts enabled
- "Processing frames... X/Y" - Frame processing progress
- "Re-encoding video for browser preview..." - H.264 re-encoding stage
- "Processing completed" - Job finished successfully

### `GET /download/{job_id}`
Download processed video.

**Query Parameters:**
- `attachment` (optional): If `true`, forces download; if `false` or omitted, inline preview

**Response:**
- Content-Type: `video/mp4`
- Content-Disposition: Properly formatted with filename `processed_{job_id}.mp4`
- Accept-Ranges: `bytes` (supports HTTP Range requests)
- Automatically re-encoded to H.264/AAC for browser compatibility

### `GET /logs/{job_id}`
Get detailed prediction logs for a completed job.

**Returns:** JSON file containing all detection results with bounding boxes, class labels, confidence scores, timestamps, ADAS alerts, and statistics.

**Log Location:** `backend/logs/{job_id}_output_predictions.json`

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

