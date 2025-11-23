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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
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
    "checkpoint_path": None,
    "pipeline_type": None
}

# Processing jobs
processing_jobs = {}


class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "error"
    progress: float
    message: str
    output_file: Optional[str] = None


def load_model_once(checkpoint_path: str, config_path: Optional[str] = None, pipeline_type: str = "original"):
    """Load model once and cache it."""
    if (_model_cache["model"] is None or 
        _model_cache["checkpoint_path"] != checkpoint_path or
        _model_cache.get("pipeline_type") != pipeline_type):
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Pipeline type: {pipeline_type}")
        
        # Determine config path based on pipeline type
        if config_path is None:
            if pipeline_type == "saliency":
                config_path = str(backend_path / "configs" / "d2city_saliency_enhanced_rtdetr.yml")
            else:
                config_path = str(backend_path / "configs" / "d2city_rtdetr.yml")
            
            if not os.path.exists(config_path):
                print(f"Warning: Config not found at {config_path}, using default")
                config_path = None
        
        # Import load_model from inference script
        sys.path.insert(0, str(backend_path / "scripts"))
        from inference import load_model as load_inference_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, cfg = load_inference_model(checkpoint_path, config_path, device)
        
        _model_cache["model"] = model
        _model_cache["config"] = cfg
        _model_cache["checkpoint_path"] = checkpoint_path
        _model_cache["pipeline_type"] = pipeline_type
        print("✓ Model loaded and cached")
    
    return _model_cache["model"], _model_cache["config"]


def process_video_task(job_id: str, input_path: str, checkpoint_path: str, conf_threshold: float = 0.5, pipeline_type: str = "original"):
    """Background task to process video."""
    try:
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = f"Loading model ({pipeline_type} pipeline)..."
        
        # Load model with pipeline-specific config
        model, cfg = load_model_once(checkpoint_path, pipeline_type=pipeline_type)
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
    checkpoint_path: Optional[str] = Form(None),
    conf_threshold: float = Form(0.5),
    pipeline_type: str = Form("original")
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
    
    # Determine checkpoint path based on pipeline type
    if checkpoint_path is None:
        # Try to find pipeline-specific checkpoint first
        output_dir = backend_path / "output"
        
        if pipeline_type == "saliency":
            # Look for saliency-enhanced checkpoint
            saliency_checkpoint_dir = output_dir / "d2city_saliency_enhanced_rtdetr_r50vd"
            if saliency_checkpoint_dir.exists():
                checkpoints = list(saliency_checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[0])
                    print(f"Using saliency-enhanced checkpoint: {checkpoint_path}")
        
        if checkpoint_path is None:
            # Fallback to original checkpoint
            original_checkpoint_dir = output_dir / "d2city_rtdetr_r50vd"
            if original_checkpoint_dir.exists():
                checkpoints = list(original_checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[0])
                    print(f"Using original checkpoint: {checkpoint_path}")
        
        # Final fallback: checkpoints directory
        if checkpoint_path is None:
            checkpoint_dir = backend_path / "checkpoints"
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            if checkpoints:
                checkpoint_path = str(checkpoints[0])
                print(f"Using pretrained checkpoint: {checkpoint_path}")
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No checkpoint found for pipeline '{pipeline_type}'. "
                           f"Please ensure model is trained or specify checkpoint_path."
                )
    
    # Validate pipeline type
    if pipeline_type not in ["original", "saliency"]:
        raise HTTPException(status_code=400, detail="Invalid pipeline_type. Must be 'original' or 'saliency'")
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": f"Video uploaded ({pipeline_type} pipeline), waiting to process...",
        "output_file": None,
        "pipeline_type": pipeline_type
    }
    
    print(f"Job {job_id}: Pipeline type = {pipeline_type}, Checkpoint = {checkpoint_path}")
    
    # Start background processing
    background_tasks.add_task(
        process_video_task,
        job_id=job_id,
        input_path=str(input_path),
        checkpoint_path=checkpoint_path,
        conf_threshold=conf_threshold,
        pipeline_type=pipeline_type
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

