# Driver Assistance System using RT-DETR

A comprehensive driver assistance system implementation using RT-DETR (Real-Time Detection Transformer) fine-tuned on the D2-City dataset. The system includes a complete training pipeline, inference scripts, and a web-based frontend with FastAPI backend for processing dashcam videos.

## Features

- **D2-City Dataset Support**: Fine-tuned on D2-City video dataset
- **RT-DETR Integration**: Uses RT-DETR's native PyTorch model implementation and training infrastructure
- **Complete Pipeline**: Training, fine-tuning, inference, and evaluation
- **Model Establishment**: Properly establishes model architecture from config, then loads pretrained weights
- **Web Interface**: Minimalistic dark-themed Next.js frontend with FastAPI backend
- **Video Processing**: Upload dashcam videos and get processed videos with bounding boxes

## RT-DETR Model Implementation

The system includes a **standalone PyTorch-based RT-DETR model implementation** (copied from `rtdetr_pytorch` for independence):

- **Model Class**: `RTDETR` (PyTorch `nn.Module`) located in `backend/src/rtdetr/zoo/rtdetr/rtdetr.py`
- **Architecture**: 
  - Backbone: ResNet/DLA/RegNet (configurable)
  - Encoder: HybridEncoder
  - Decoder: RTDETRTransformer
- **Checkpoints**: Pretrained weights are loaded into the established architecture
- **Registration**: Uses RT-DETR's `@register` decorator for dependency injection
- **Standalone**: No dependency on external `rtdetr_pytorch` repository - all code is self-contained

The model is established from YAML config files, then pretrained weights are loaded for fine-tuning or inference.

## Directory Structure

```
driver-assistance-system-using-RT-DETR/
├── backend/                    # All backend code
│   ├── api/                   # FastAPI service
│   │   ├── main.py            # FastAPI application
│   │   ├── run_server.py      # Server startup script
│   │   ├── requirements.txt   # Python dependencies
│   │   └── README.md          # API documentation
│   ├── checkpoints/           # Pretrained model checkpoints
│   ├── configs/               # Configuration files
│   │   └── d2city_rtdetr.yml  # D2-City fine-tuning config
│   ├── scripts/               # All scripts
│   │   ├── finetuning.py     # Fine-tune on D2-City (PRIMARY)
│   │   ├── inference.py      # Run inference (PRIMARY)
│   │   ├── process_video.py  # Process videos with bboxes
│   │   ├── evaluate.py        # Calculate mAP
│   │   ├── train.py          # Training script (reference)
│   │   └── test.py           # Test script (reference)
│   ├── src/
│   │   └── datasets/
│   │       ├── d2city_dataset_rtdetr.py  # D2-City dataset loader
│   │       └── register_rtdetr.py        # Register with RT-DETR
│   ├── uploads/               # Uploaded videos
│   └── outputs/               # Processed videos
├── frontend/                  # Next.js frontend
│   ├── app/
│   │   ├── page.tsx          # Main page component
│   │   ├── layout.tsx        # Root layout
│   │   └── globals.css       # Dark theme styles
│   ├── package.json          # Node dependencies
│   └── tsconfig.json         # TypeScript config
├── start_backend.sh          # Backend startup script
├── start_frontend.sh         # Frontend startup script
└── requirements.txt          # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 18+ with npm
- RT-DETR checkpoint in `backend/checkpoints/`
- CUDA (optional, for GPU acceleration)

### 1. Install Python Dependencies

```bash
cd driver-assistance-system-using-RT-DETR
pip install -r requirements.txt
```

### 2. Install Backend API Dependencies

```bash
cd backend/api
pip install -r requirements.txt
```

### 3. RT-DETR is Included

The RT-DETR model implementation is included in `backend/src/rtdetr/` - no need to install external RT-DETR repository.

### 4. Install Frontend Dependencies

```bash
cd ../driver-assistance-system-using-RT-DETR/frontend
npm install
```

## Quick Start

### Start the System

**Terminal 1 - Backend:**
```bash
./start_backend.sh
# or
cd backend
python api/run_server.py
```

Backend will run at: `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
# or
cd frontend
npm run dev
```

Frontend will run at: `http://localhost:3000`

### Use the Web Interface

1. Open browser: `http://localhost:3000`
2. Upload a dashcam video
3. Wait for processing
4. View processed video with bounding boxes
5. Download if needed

## Usage

### Fine-Tuning on D2-City

```bash
# Register dataset first
python -c "import sys; sys.path.insert(0, 'backend/src/datasets'); from register_rtdetr import *"

# Fine-tune
python backend/scripts/finetuning.py \
    --config backend/configs/d2city_rtdetr.yml \
    --pretrained-checkpoint backend/checkpoints/rtdetr_r101vd_6x_coco.pth \
    --amp
```

### Inference on Images

```bash
python backend/scripts/inference.py \
    --checkpoint backend/output/d2city_rtdetr_r50vd/checkpoint.pth \
    --input image.jpg \
    --output results.json
```

### Process Video (CLI)

```bash
python backend/scripts/process_video.py \
    --checkpoint backend/output/d2city_rtdetr_r50vd/checkpoint.pth \
    --input video.mp4 \
    --output output.mp4 \
    --save-predictions
```

### Evaluate Model

```bash
python backend/scripts/evaluate.py \
    --config backend/configs/d2city_rtdetr.yml \
    --checkpoint backend/output/d2city_rtdetr_r50vd/checkpoint.pth \
    --output-dir evaluation_results
```

## Web API

### Backend API Endpoints

The FastAPI backend provides the following endpoints:

#### `POST /upload`
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

#### `GET /status/{job_id}`
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

#### `GET /download/{job_id}`
Download processed video.

**Returns:** Video file (MP4)

#### `GET /health`
Health check with model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

### Frontend Features

- **Minimalistic Dark Theme**: Pure black (#000000) and white (#ffffff) color scheme
- **Video Upload**: Drag & drop or click to select
- **Real-time Progress**: Status updates with progress bar
- **Video Preview**: Embedded video player with bounding boxes
- **Download**: Direct download link for processed videos
- **Error Handling**: User-friendly error messages

## Model Establishment Flow

The system follows RT-DETR's pattern for model establishment:

1. **Establish Architecture**: Model structure (backbone, encoder, decoder) created from YAML config
2. **Load Pretrained Weights**: Weights loaded into established architecture
3. **Fine-tune/Inference**: Model ready for training or inference

### Example Flow

```python
# Step 1: Load config (establishes model architecture)
cfg = YAMLConfig(config_path, tuning=pretrained_path)

# Step 2: Model architecture is created via cfg.model property
# This instantiates: RTDETR(backbone, encoder, decoder)

# Step 3: Create solver (loads weights in setup())
solver = TASKS[cfg.yaml_cfg['task']](cfg)

# Step 4: Training/Inference
solver.fit()  # or solver.val()
```

## Dataset Setup

Place D2-City MP4 video files in:
```
backend/datasets/d2_city/
  ├── video1.mp4
  ├── video2.mp4
  └── ...
```

## Scripts

### Primary Scripts

- **`finetuning.py`**: Primary fine-tuning script (uses RT-DETR solver)
- **`inference.py`**: Primary inference script (returns bbox, labels, scores, centroids)
- **`process_video.py`**: Process videos with bounding boxes
- **`evaluate.py`**: Calculate mAP and generate reports

### Reference Scripts

- **`train.py`**: Reference training script (inspired by RT-DETR)
- **`test.py`**: Reference test script (inspired by RT-DETR)

## Model Architecture

The model architecture is established from RT-DETR config files:

- **Backbone**: ResNet/DLA/RegNet (defined in config)
- **Encoder**: Hybrid Encoder
- **Decoder**: Transformer Decoder

Pretrained weights are loaded into this architecture for fine-tuning.

## Outputs

- **Checkpoints**: Saved in `backend/output/d2city_rtdetr_r50vd/`
- **Evaluation**: mAP charts and reports in `evaluation_results/`
- **Videos**: Annotated videos with bounding boxes
- **Predictions**: JSON/TXT files with detections (bbox, class, score, centroid)

## Technical Details

### Backend
- **Framework**: FastAPI
- **Processing**: Background tasks with asyncio
- **Model**: RT-DETR PyTorch implementation (cached after first load)
- **Video**: OpenCV for processing
- **Storage**: Local file system

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: CSS Modules with global styles
- **HTTP Client**: Axios
- **State Management**: React hooks

## Dependencies

### Backend
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `python-multipart>=0.0.6`
- `pydantic>=2.0.0`
- PyTorch (from RT-DETR)

### Frontend
- `next@14.0.0`
- `react@^18.2.0`
- `axios@^1.6.0`
- `typescript@^5.0.0`

## Troubleshooting

### Backend Issues
- **Backend won't start**: Check that checkpoint exists in `backend/checkpoints/`
- **Model not loading**: Verify RT-DETR is in the correct path (`../rtdetr_pytorch`)
- **CUDA errors**: Ensure CUDA is properly installed if using GPU
- **Import errors**: Check that all paths are correctly set in scripts

### Frontend Issues
- **Frontend can't connect**: Ensure backend is running on port 8000
- **CORS errors**: Check `backend/api/main.py` CORS settings
- **API errors**: Verify `NEXT_PUBLIC_API_URL` in `.env.local` (default: `http://localhost:8000`)

### Training/Inference Issues
- **Checkpoint not found**: Ensure checkpoint path is correct
- **Config errors**: Verify YAML config file syntax
- **Dataset errors**: Check dataset path and structure

## Next Steps

1. **Fine-tune on Saliency Dataset**: Once saliency-based D2-City dataset is ready
2. **Deploy**: Consider Docker containers for production
3. **Optimize**: Add video compression, caching, CDN
4. **Enhance**: Add batch processing, user accounts, history

## Notes

- Backend processes videos in background (non-blocking)
- Model is cached after first load (faster subsequent requests)
- Frontend polls status every 2 seconds
- Videos are stored locally (consider cloud storage for production)
- CORS is configured for localhost:3000

## License

This project uses RT-DETR, which is licensed under Apache 2.0.
