# Driver Assistance System - Frontend

Minimalistic, modern, dark-themed Next.js frontend for the Driver Assistance System.

## Features

- **Dark Theme**: Black and white minimalist design
- **Video Upload**: Upload dashcam videos for processing
- **Pipeline Selection**: Choose between Original D2-City and Saliency-Enhanced pipelines
- **Real-time Status**: Track processing progress with live updates
- **System Logs**: Terminal-style log output showing processing progress and ADAS status
- **Video Preview**: View processed videos with bounding boxes and ADAS alerts
- **Download**: Download processed videos (properly formatted as .mp4)

## Setup

1. **Install dependencies**:
```bash
cd frontend
npm install
```

2. **Set environment variable** (optional):
```bash
# Create .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. **Run development server**:
```bash
npm run dev
```

4. **Open browser**:
```
http://localhost:3000
```

## Usage

1. Select a dashcam video file
2. Click "Upload & Process"
3. Wait for processing to complete
4. View the processed video with bounding boxes
5. Download the result

## API Integration

The frontend communicates with the FastAPI backend at `http://localhost:8000`:

- `POST /upload` - Upload video with pipeline type selection
- `GET /status/{job_id}` - Get processing status (with automatic job recovery)
- `GET /download/{job_id}` - Download processed video with proper Content-Disposition headers
- `GET /logs/{job_id}` - Retrieve detailed prediction logs (JSON format)

