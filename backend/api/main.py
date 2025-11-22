"""
FastAPI Backend Service
Handles video upload, processing, and download
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import RT-DETR from local copy
from src.rtdetr.core import YAMLConfig
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Driver Assistance System API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = backend_path / "uploads"
OUTPUT_DIR = backend_path / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model cache
_model_cache = {
    "model": None,
    "config": None,
    "checkpoint_path": None
}

# Processing jobs
processing_jobs = {}


class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "error"
    progress: float
    message: str
    output_file: Optional[str] = None


def load_model_once(checkpoint_path: str, config_path: Optional[str] = None):
    """Load model once and cache it."""
    if (_model_cache["model"] is None or 
        _model_cache["checkpoint_path"] != checkpoint_path):
        
        print(f"Loading model from: {checkpoint_path}")
        # Import load_model from inference script
        sys.path.insert(0, str(backend_path / "scripts"))
        from inference import load_model as load_inference_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, cfg = load_inference_model(checkpoint_path, config_path, device)
        
        _model_cache["model"] = model
        _model_cache["config"] = cfg
        _model_cache["checkpoint_path"] = checkpoint_path
        print("✓ Model loaded and cached")
    
    return _model_cache["model"], _model_cache["config"]


def process_video_task(job_id: str, input_path: str, checkpoint_path: str, conf_threshold: float = 0.5):
    """Background task to process video."""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Loading model..."
        
        # Load model
        model, cfg = load_model_once(checkpoint_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        processing_jobs[job_id]["progress"] = 0.2
        processing_jobs[job_id]["message"] = "Processing video..."
        
        # Generate output path
        output_filename = f"{job_id}_output.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        # Import process_video function from scripts
        sys.path.insert(0, str(backend_path / "scripts"))
        from process_video import process_video
        
        # Process video (this handles everything)
        process_video(
            model=model,
            video_path=input_path,
            output_path=str(output_path),
            device=device,
            conf_threshold=conf_threshold,
            save_predictions=False,  # Skip predictions for API
            output_format='json'
        )
        
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "Processing completed"
        processing_jobs[job_id]["output_file"] = output_filename
        
    except Exception as e:
        import traceback
        processing_jobs[job_id]["status"] = "error"
        processing_jobs[job_id]["message"] = f"Error: {str(e)}"
        print(f"Error processing video: {e}")
        traceback.print_exc()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Driver Assistance System API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check with model status."""
    model_loaded = _model_cache["model"] is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    checkpoint_path: Optional[str] = None,
    conf_threshold: float = 0.5
):
    """
    Upload video for processing.
    
    Returns job_id for tracking processing status.
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix
    if file_extension not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    input_filename = f"{job_id}_input{file_extension}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_dir = backend_path / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            checkpoint_path = str(checkpoints[0])
        else:
            raise HTTPException(
                status_code=404,
                detail="No checkpoint found. Please specify checkpoint_path or place .pth file in backend/checkpoints/"
            )
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Video uploaded, waiting to process...",
        "output_file": None
    }
    
    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id=job_id,
        input_path=str(input_path),
        checkpoint_path=checkpoint_path,
        conf_threshold=conf_threshold
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Video uploaded successfully"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status for a job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return ProcessingStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        output_file=job.get("output_file")
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download processed video."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video not ready yet")
    
    output_file = job.get("output_file")
    if not output_file:
        raise HTTPException(status_code=404, detail="Output file not found")
    
    file_path = OUTPUT_DIR / output_file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")
    
    return FileResponse(
        path=str(file_path),
        filename=f"processed_{job_id}.mp4",
        media_type="video/mp4"
    )


@app.get("/jobs")
async def list_jobs():
    """List all processing jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"]
            }
            for job_id, job in processing_jobs.items()
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

