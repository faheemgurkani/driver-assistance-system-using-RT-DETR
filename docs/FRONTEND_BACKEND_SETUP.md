# Frontend and Backend Setup Guide

This document provides a comprehensive guide to setting up and understanding the relationship between the frontend and backend components of the Driver Assistance System using RT-DETR.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [API Endpoints](#api-endpoints)
5. [Communication Flow](#communication-flow)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## System Architecture

The Driver Assistance System consists of two main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│  - React-based UI                                            │
│  - Video upload interface                                    │
│  - Real-time status tracking                                 │
│  - Processed video preview                                    │
│  Port: 3000                                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
                       │ (CORS enabled)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (FastAPI)                               │
│  - FastAPI REST API                                          │
│  - Video processing service                                  │
│  - RT-DETR model inference                                   │
│  - Job management                                            │
│  Port: 8000                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Relationship

- **Frontend**: User-facing web interface built with Next.js 14, React, and TypeScript
- **Backend**: FastAPI service that handles video processing, model inference, and file management
- **Communication**: RESTful API with JSON responses and multipart form data for file uploads
- **CORS**: Configured to allow requests from `http://localhost:3000` and `http://127.0.0.1:3000`

---

## Backend Setup

### Prerequisites

- Python 3.8+
- PyTorch (CUDA or MPS, depending on hardware)
- All backend dependencies from `backend/requirements.txt`
- **COCO pretrained RT-DETR ResNet-101-VD checkpoint** (`backend/checkpoints/rtdetr_r101vd_6x_coco.pth`)

### Installation Steps

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install API-specific dependencies**:
   ```bash
   cd api
   pip install -r requirements.txt
   ```

4. **Verify model checkpoints are available**:
   - Checkpoints should be in `backend/checkpoints/` or `backend/output/`
   - Default checkpoint: `rtdetr_r101vd_6x_coco.pth`

### Running the Backend

**Option 1: Using the run script**:
```bash
cd backend/api
python run_server.py
```

**Option 2: Using uvicorn directly**:
```bash
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Option 3: Using the project start script (run from repo root)**:
```bash
bash scripts/start_backend.sh
```

The backend will be available at:
- **API Base URL**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

### Backend Configuration

The backend uses the following default paths:
- **Upload Directory**: `backend/uploads/` (created automatically)
- **Output Directory**: `backend/outputs/` (created automatically)
- **Logs Directory**: `backend/logs/` (created automatically)
- **Model Checkpoints**: 
  - `backend/checkpoints/` (pretrained models)
  - `backend/output/` (fine-tuned models)

### Backend Features

- **Model Caching**: Models are loaded once and cached in memory for faster subsequent requests
- **Background Processing**: Video processing runs asynchronously using FastAPI BackgroundTasks
- **Job Management**: Each video upload gets a unique job ID for tracking
- **Pipeline Support**: Supports both "original" and "saliency-enhanced" processing pipelines
- **ADAS Alerts**: Blind spot detection and collision warning integration with visual overlays
- **Automatic Re-encoding**: Processed MP4s are re-encoded to H.264/AAC with `faststart` for guaranteed browser playback
- **HTTP Range Streaming**: `/download/{job_id}` honors `Range` headers so previews can begin before the file finishes downloading
- **Progress Mirroring**: Frame counts, re-encoding status, and ADAS status are written to the job object and surface in the frontend System Log
- **Automatic Prediction Logging**: Detailed JSON logs are automatically generated for every completed video processing job, containing all detection results with bounding boxes, class labels, confidence scores, timestamps, ADAS alerts, and statistics
- **Job Recovery**: Automatic recovery of completed jobs from filesystem after server reloads
- **Enhanced Text Rendering**: Large, high-contrast text labels (black on green boxes, white on colored boxes)

### ADAS Modules

The ADAS functionality is implemented as modular components in `backend/Alerts/modules/`:

- **Blind Spot Detection** (`blind_spot/blind_spot.py`): Detects vehicles in left (0-25% width) and right (75-100% width) blind spot zones at 60-100% frame height
- **Collision Warning** (`collision/collision_warning.py`): Estimates distance to vehicles in frontal lane (30-70% width) and classifies collision risk as HIGH/MEDIUM/LOW

These modules are automatically imported during video processing. If unavailable, the system gracefully falls back to standard detections.

### Notebooks

Jupyter notebooks for dataset preparation are located in `backend/src/notebooks/`:

- **`preprocessing_training_data.ipynb`**: Processes raw training videos and XML annotations from D2-City dataset
- **`saliency_module.ipynb`**: Generates saliency masks and creates saliency-enhanced frames using Salience-DETR model

These notebooks are used during dataset preparation before training the saliency-enhanced pipeline.

---

## Frontend Setup

### Prerequisites

- Node.js 18+ and npm
- Modern web browser

### Installation Steps

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment variables** (optional):
   Create `.env.local` file:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
   
   If not set, defaults to `http://localhost:8000`

### Running the Frontend

**Option 1: Development mode**:
```bash
cd frontend
npm run dev
```

**Option 2: Using the project start script (run from repo root)**:
```bash
bash scripts/start_frontend.sh
```

**Option 3: Production build**:
```bash
cd frontend
npm run build
npm start
```

The frontend will be available at:
- **URL**: `http://localhost:3000`

### Frontend Features

- **Dark Theme UI**: Minimalist black and white design
- **Video Upload**: Drag-and-drop or click to upload
- **Pipeline Selection**: Choose between "original" and "saliency-enhanced" pipelines
- **Real-time Status**: Progress tracking with live updates
- **System Logs**: Terminal-style log output showing processing progress and ADAS status
- **Video Preview**: In-browser video player for processed results with bounding boxes and ADAS alerts
- **Download**: Direct download of processed videos (properly formatted as .mp4)
- **Documentation Tab**: Built-in documentation viewer

### Video Preview & Download Flow

1. **Upload & Polling** – the frontend posts to `/upload` and polls `/status/{job_id}` every two seconds.
2. **ADAS Initialization** – if ADAS is available, status message shows "ADAS: Mapping blind-spot and collision risks per frame..."
3. **Frame Progress** – messages such as "Processing frames… 300/453" stream into the System Log.
4. **Re-encoding Stage** – after inference finishes, the backend re-encodes the MP4 to H.264/AAC and updates the status to "Re-encoding video for browser preview…".
5. **Streaming Preview** – the `<video>` element requests `/download/{job_id}` and receives `206 Partial Content` plus `Accept-Ranges: bytes`, enabling instant playback and scrubbing.
6. **Download Result** – the button calls `/download/{job_id}?attachment=1&ts=...` via a programmatic link with proper Content-Disposition header ensuring .mp4 extension.
7. **Cache Busting** – URLs include timestamps to ensure the browser loads the newest file.

If the preview ever stalls at 0:00, open DevTools → Network → select the video request and confirm:
- Status is `200` or `206`
- `Content-Type` is `video/mp4`
- `Accept-Ranges: bytes` is present

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| GET | `/` | Health check | None |
| GET | `/health` | Health check with model status | None |
| POST | `/upload` | Upload video for processing | None |
| GET | `/status/{job_id}` | Get processing status | None |
| GET | `/download/{job_id}` | Download processed video | None |
| GET | `/logs/{job_id}` | Get prediction logs (JSON) | None |
| GET | `/jobs` | List all processing jobs | None |

---

### 1. Health Check

**Endpoint**: `GET /`

**Description**: Basic health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "service": "Driver Assistance System API",
  "version": "1.0.0"
}
```

**Example**:
```bash
curl http://localhost:8000/
```

---

### 2. Health Check with Model Status

**Endpoint**: `GET /health`

**Description**: Health check that includes model loading status.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

**Response Fields**:
- `status`: Service status ("ok")
- `model_loaded`: Boolean indicating if model is loaded in cache
- `device`: Device type ("cuda", "mps", or "cpu")

**Example**:
```bash
curl http://localhost:8000/health
```

---

### 3. Upload Video

**Endpoint**: `POST /upload`

**Description**: Upload a video file for processing. Returns a job ID for tracking.

**Request Type**: `multipart/form-data`

**Form Fields**:
- `file` (required): Video file (MP4, AVI, MOV, MKV)
- `checkpoint_path` (optional): Path to model checkpoint file
- `conf_threshold` (optional): Confidence threshold (default: 0.5, range: 0.0-1.0)
- `pipeline_type` (optional): Processing pipeline type - "original" or "saliency" (default: "original")

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Video uploaded successfully"
}
```

**Response Fields**:
- `job_id`: Unique identifier for tracking the processing job
- `status`: Initial status ("pending")
- `message`: Status message

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@video.mp4" \
  -F "conf_threshold=0.5" \
  -F "pipeline_type=original"
```

**Example using Python**:
```python
import requests

files = {'file': open('video.mp4', 'rb')}
data = {
    'conf_threshold': 0.5,
    'pipeline_type': 'original'
}
response = requests.post('http://localhost:8000/upload', files=files, data=data)
print(response.json())
```

**Pipeline Types**:
- `"original"`: Uses original D2-City dataset pipeline with data loading and preprocessing
- `"saliency"`: Uses pre-processed saliency-enhanced frames (no preprocessing needed)

**Checkpoint Selection**:
If `checkpoint_path` is not provided, the backend will:
1. For "saliency" pipeline: Look for checkpoint in `backend/output/d2city_saliency_enhanced_rtdetr_r101vd/` (preferred) or `d2city_saliency_enhanced_rtdetr_r50vd/`
2. For "original" pipeline: Look for checkpoint in `backend/output/d2city_rtdetr_r101vd/` (preferred) or `d2city_rtdetr_r50vd/`
3. Fallback: Use any `.pth` file from `backend/checkpoints/`

---

### 4. Get Processing Status

**Endpoint**: `GET /status/{job_id}`

**Description**: Get the current status of a processing job.

**Path Parameters**:
- `job_id` (required): The job ID returned from `/upload`

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.65,
  "message": "Processing video...",
  "output_file": "550e8400-e29b-41d4-a716-446655440000_output.mp4"
}
```

**Response Fields**:
- `job_id`: The job identifier
- `status`: Current status - "pending", "processing", "completed", or "error"
- `progress`: Progress value (0.0 to 1.0)
- `message`: Human-readable status message (includes ADAS status when applicable)
- `output_file`: Filename of the processed video (only when status is "completed")
- `log_file`: Filename of the prediction log JSON file (only when status is "completed")

**Job Recovery**:
If a job is not found in memory (e.g., after server reload), the endpoint automatically attempts to recover it from the filesystem by checking if the output video exists. If found, the job status is restored as "completed".

**Status Values**:
- `"pending"`: Job is queued, waiting to start
- `"processing"`: Video is being processed
- `"completed"`: Processing finished successfully
- `"error"`: Processing failed

**Example**:
```bash
curl http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000
```

**Polling Recommendation**:
The frontend polls this endpoint every 2 seconds while status is "pending" or "processing".

---

### 5. Download Processed Video

**Endpoint**: `GET /download/{job_id}`

**Description**: Download the processed video file.

**Path Parameters**:
- `job_id` (required): The job ID returned from `/upload`

**Query Parameters**:
- `attachment` (optional): If `true`, forces download; if `false` or omitted, inline preview

**Response**: 
- Content-Type: `video/mp4`
- Content-Disposition: Properly formatted with filename `processed_{job_id}.mp4` and UTF-8 encoding
- Accept-Ranges: `bytes` (supports HTTP Range requests)
- File download with correct .mp4 extension

**Error Responses**:
- `404`: Job not found
- `400`: Video not ready yet (status is not "completed")

**Example**:
```bash
curl "http://localhost:8000/download/550e8400-e29b-41d4-a716-446655440000" \
  -o output.mp4
```

**Example in browser**:
```html
<a href="http://localhost:8000/download/550e8400-e29b-41d4-a716-446655440000" download>
  Download Video
</a>
```

---

### 6. Get Prediction Logs

**Endpoint**: `GET /logs/{job_id}`

**Description**: Retrieve detailed JSON prediction logs for a completed video processing job. The logs contain comprehensive information about all detections, including bounding box coordinates, class labels, confidence scores, timestamps, and statistics.

**Path Parameters**:
- `job_id` (required): The job ID returned from `/upload`

**Response**: 
- Content-Type: `application/json`
- Complete prediction log with metadata, statistics, and per-frame detections

**Error Responses**:
- `404`: Job not found or log file not available
- `400`: Job not completed yet (status is not "completed")

**Response Structure**:
```json
{
  "metadata": {
    "processing_timestamp": "2025-11-28T21:20:12.415675",
    "completion_timestamp": "2025-11-28T21:23:22.185058",
    "input_video": {
      "path": "/path/to/input.mp4",
      "filename": "input.mp4",
      "resolution": {
        "width": 3840,
        "height": 2160
      },
      "fps": 29.97,
      "total_frames": 453,
      "duration_seconds": 15.12
    },
    "output_video": {
      "path": "/path/to/output.mp4",
      "filename": "output.mp4"
    },
    "model": {
      "device": "mps",
      "confidence_threshold": 0.5,
      "checkpoint_path": "/path/to/checkpoint.pth",
      "config_path": "/path/to/config.yml",
      "pipeline_type": "saliency"
    }
  },
  "statistics": {
    "total_frames_processed": 453,
    "total_detections": 8133,
    "frames_with_detections": 453,
    "average_detections_per_frame": 17.95,
    "class_distribution": {
      "car": 5959,
      "bus": 280,
      "person": 1829,
      "truck": 62,
      "motorcycle": 3
    }
  },
  "predictions": [
    {
      "frame_number": 0,
      "timestamp_seconds": 0.0,
      "timestamp_formatted": "00:00.00",
      "num_detections": 18,
      "detections": [
        {
          "bbox": {
            "x1": 2081.36,
            "y1": 1708.63,
            "x2": 2845.73,
            "y2": 2159.42,
            "format": "xyxy",
            "width": 764.38,
            "height": 450.79,
            "area": 344573.37
          },
          "bbox_normalized": {
            "x1": 0.54202,
            "y1": 0.791031,
            "x2": 0.741077,
            "y2": 0.99973
          },
          "centroid": {
            "x": 2463.55,
            "y": 1934.02
          },
          "centroid_normalized": {
            "x": 0.641549,
            "y": 0.895381
          },
          "class": {
            "id": 2,
            "name": "car"
          },
          "confidence": 0.949279
        }
      ]
    }
  ]
}
```

**Log File Location**:
- Log files are saved in `backend/logs/` directory
- Filename format: `{job_id}_output_predictions.json`
- Logs are automatically generated for every completed video processing job
- Logs include ADAS alert information (blind spot and collision warnings) per frame

**Use Cases**:
- Analysis of detection patterns across video frames
- ADAS alert analysis and statistics
- Performance evaluation and statistics
- Debugging detection issues
- Exporting detection data for further processing
- Generating reports on detected objects and safety alerts

**Example**:
```bash
curl http://localhost:8000/logs/550e8400-e29b-41d4-a716-446655440000
```

**Example with jq** (pretty print):
```bash
curl http://localhost:8000/logs/550e8400-e29b-41d4-a716-446655440000 | jq '.statistics'
```

---

### 7. List All Jobs

**Endpoint**: `GET /jobs`

**Description**: Get a list of all processing jobs and their statuses.

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "progress": 1.0,
      "message": "Processing completed"
    },
    {
      "job_id": "660e8400-e29b-41d4-a716-446655440001",
      "status": "processing",
      "progress": 0.5,
      "message": "Processing video..."
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/jobs
```

---

## Communication Flow

### Typical Workflow

```
┌──────────┐                    ┌──────────┐
│ Frontend │                    │ Backend  │
└────┬─────┘                    └────┬─────┘
     │                               │
     │  1. POST /upload              │
     │  (video file + params)        │
     ├──────────────────────────────>│
     │                               │
     │  2. Response: {job_id, ...}   │
     │<──────────────────────────────┤
     │                               │
     │  3. GET /status/{job_id}      │
     │  (poll every 2 seconds)       │
     ├──────────────────────────────>│
     │                               │
     │  4. Response: {status,        │
     │     progress, message}        │
     │<──────────────────────────────┤
     │                               │
     │  (Repeat step 3-4 until       │
     │   status = "completed")       │
     │                               │
     │  5. GET /download/{job_id}    │
     ├──────────────────────────────>│
     │                               │
     │  6. Response: video file      │
     │<──────────────────────────────┤
     │                               │
```

### Detailed Flow

1. **User uploads video**:
   - Frontend sends `POST /upload` with video file and parameters
   - Backend saves file, creates job, starts background processing
   - Backend returns `job_id`

2. **Status polling**:
   - Frontend polls `GET /status/{job_id}` every 2 seconds
   - Backend returns current status, progress, and message
   - Frontend updates UI with progress bar and logs

3. **Processing completion**:
   - When status becomes "completed", frontend stops polling
   - Backend automatically generates detailed JSON prediction logs with ADAS alert information
   - Log file is saved as `{job_id}_output_predictions.json` in `backend/logs/`
   - Video is automatically re-encoded to H.264/AAC for browser compatibility
   - Frontend displays video preview using `GET /download/{job_id}` URL with HTTP Range support
   - User can download the processed video (properly formatted as .mp4)
   - Prediction logs can be retrieved via `GET /logs/{job_id}` endpoint

---

## Configuration

### Backend Configuration

**CORS Settings** (`backend/api/main.py`):
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Directories**:
- Upload directory: `backend/uploads/`
- Output directory: `backend/outputs/`
- Checkpoints: `backend/checkpoints/` or `backend/output/`

**Server Settings**:
- Host: `0.0.0.0` (all interfaces)
- Port: `8000`
- Reload: Enabled in development

### Frontend Configuration

**API URL** (`frontend/app/page.tsx`):
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
```

**Environment Variables** (`.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Polling Interval**: 2000ms (2 seconds)

---

## Troubleshooting

### Backend Issues

**Problem**: Backend won't start
- **Solution**: Check if port 8000 is already in use
- **Solution**: Verify all dependencies are installed: `pip install -r backend/api/requirements.txt`

**Problem**: Model not loading
- **Solution**: Ensure checkpoint file exists in `backend/checkpoints/` or `backend/output/`
- **Solution**: Check model file path in error messages

**Problem**: CORS errors
- **Solution**: Verify frontend URL is in `allow_origins` list
- **Solution**: Check that backend is running on port 8000

### Frontend Issues

**Problem**: Cannot connect to backend
- **Solution**: Verify backend is running: `curl http://localhost:8000/health`
- **Solution**: Check `NEXT_PUBLIC_API_URL` environment variable
- **Solution**: Ensure CORS is properly configured on backend

**Problem**: Upload fails
- **Solution**: Check file size (backend may have limits)
- **Solution**: Verify file format is supported (MP4, AVI, MOV, MKV)
- **Solution**: Check browser console for error messages

**Problem**: Status polling not working
- **Solution**: Check browser console for network errors
- **Solution**: Verify job_id is correct
- **Solution**: Check backend logs for processing errors

### Common Error Messages

**"Job not found"**:
- Job ID is incorrect or job was cleared from memory
- Solution: Upload video again to get a new job_id

**"Video not ready yet"**:
- Processing is still in progress
- Solution: Continue polling status endpoint

**"No checkpoint found"**:
- Model checkpoint file is missing
- Solution: Ensure checkpoint exists in expected directory or specify `checkpoint_path` in upload

**"Invalid video format"**:
- Uploaded file is not a supported video format
- Solution: Use MP4, AVI, MOV, or MKV format

---

## Additional Resources

- **Backend API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Backend README**: `backend/api/README.md`
- **Frontend README**: `frontend/README.md`
- **Project README**: `README.md`

---

## Quick Start Commands

**Start Backend**:
```bash
cd backend/api
python run_server.py
```

**Start Frontend**:
```bash
cd frontend
npm run dev
```

**Test Backend**:
```bash
curl http://localhost:8000/health
```

**Test Upload** (example):
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@test_video.mp4" \
  -F "pipeline_type=original" \
  -F "conf_threshold=0.5"
```

---

*Last Updated: 2024*

