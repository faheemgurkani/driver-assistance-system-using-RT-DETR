# Driver Assistance System - Frontend

Minimalistic, modern, dark-themed Next.js frontend for the Driver Assistance System.

## Features

- **Dark Theme**: Black and white minimalist design
- **Video Upload**: Upload dashcam videos for processing
- **Real-time Status**: Track processing progress
- **Video Preview**: View processed videos with bounding boxes
- **Download**: Download processed videos

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

- `POST /upload` - Upload video
- `GET /status/{job_id}` - Get processing status
- `GET /download/{job_id}` - Download processed video

