# Driver Assistance System using RT-DETR

A comprehensive driver assistance system implementation using RT-DETR (Real-Time Detection Transformer) fine-tuned on the D2-City dataset. The system includes a complete training pipeline, inference scripts, and a web-based frontend with FastAPI backend for processing dashcam videos.

## Features

- **Dual Pipeline Support**: Original D2-City (video extraction) and Saliency-Enhanced (pre-processed frames)
- **ResNet-101 Backbone**: All transfer-learning and inference paths default to the COCO-pretrained **RT-DETR R101VD** checkpoint (`rtdetr_r101vd_6x_coco.pth`)
- **Complete RT-DETR Stack**: Standalone PyTorch implementation copied from RT-DETR for full control
- **End-to-End Pipeline**: Training/fine-tuning scripts, FastAPI inference service, and Next.js frontend
- **Inline Playback**: Backend re-encodes outputs to H.264/AAC and serves HTTP range responses for instant browser preview
- **Mirrored Progress Logs**: Backend frame counts and re-encoding status show up inside the frontend terminal log

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
│   │   ├── d2city_rtdetr.yml  # Original D2-City config (with preprocessing)
│   │   ├── d2city_saliency_enhanced_rtdetr.yml  # Saliency-enhanced config
│   │   └── rtdetr_base/       # RT-DETR base configs
│   ├── scripts/               # All scripts
│   │   ├── finetuning.py     # Fine-tune on D2-City (PRIMARY)
│   │   ├── inference.py      # Run inference (PRIMARY)
│   │   ├── process_video.py  # Process videos with bboxes (used by API)
│   │   ├── train.py          # Training script (reference, supports both pipelines)
│   │   └── test.py           # Test script (reference, supports both pipelines)
│   ├── src/
│   │   ├── datasets/
│   │   │   ├── d2city_dataset_rtdetr.py  # Original D2-City dataset loader
│   │   │   ├── saliency_enhanced_d2city_dataset_rtdetr.py  # Saliency-enhanced loader
│   │   │   └── register_rtdetr.py  # Register datasets with RT-DETR
│   │   ├── rtdetr/            # RT-DETR model implementation (standalone)
│   │   └── utils/
│   │       └── saliency_integration.py  # Saliency enhancement utilities
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
- **RT-DETR R101VD checkpoint** (`rtdetr_r101vd_6x_coco.pth`) placed in `backend/checkpoints/`
- CUDA (optional) or Apple Silicon (MPS) for acceleration

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
bash scripts/start_backend.sh
# or from backend/api
# python run_server.py
```

Backend will run at: `http://localhost:8000`

**Terminal 2 - Frontend:**

```bash
bash scripts/start_frontend.sh
# or (from frontend/)
# npm run dev
```

Frontend will run at: `http://localhost:3000`

### Use the Web Interface

1. Open browser: `http://localhost:3000`
2. Select pipeline type from dropdown:
   - **Original D2-City**: Uses original dataset with data loading and preprocessing
   - **Saliency-Enhanced**: Uses pre-processed saliency-enhanced frames (no preprocessing)
3. Upload a dashcam video
4. Wait for processing
5. View the processed video inline (FastAPI streams the MP4 with range support)
6. Use the **Download Result** button to force a browser download (`attachment=1`)
7. Monitor the terminal-style log to see frame progress and the final re-encoding status

### Inline Preview & Download Flow

- The backend always re-encodes processed videos to **H.264/AAC** with `faststart` so browsers can decode them.
- `GET /download/{job_id}` supports HTTP range requests; the `<video>` tag loads metadata almost instantly and starts playing.
- The same endpoint accepts `?attachment=1` to force a download—used by the frontend button.

## Usage

### Fine-Tuning on D2-City

> **Default setup**: All production runs use the **ResNet-101-VD COCO checkpoint** with the saliency-enhanced dataset (`configs/d2city_saliency_enhanced_rtdetr_r101vd.yml`). The legacy ResNet-50 config is kept only for backwards compatibility.

#### Recommended: Saliency-Enhanced Dataset (ResNet-101)

```bash
# Register dataset (one-time)
python -c "import sys; sys.path.insert(0, 'backend/src/datasets'); from register_rtdetr import *"

python backend/scripts/finetuning.py \
    --config backend/configs/d2city_saliency_enhanced_rtdetr_r101vd.yml \
    --pretrained-checkpoint backend/checkpoints/rtdetr_r101vd_6x_coco.pth \
    --amp
```

#### Legacy: Original D2-City Videos (ResNet-50 config)

```bash
# Only needed if you want to re-run the legacy pipeline
python backend/scripts/finetuning.py \
    --config backend/configs/d2city_rtdetr.yml \
    --pretrained-checkpoint backend/checkpoints/rtdetr_r101vd_6x_coco.pth \
    --amp
```

### Training (Alternative to Fine-tuning)

```bash
# Original D2-City
python backend/scripts/train.py \
    --config backend/configs/d2city_rtdetr.yml \
    --tuning backend/checkpoints/rtdetr_r101vd_6x_coco.pth

# Saliency-Enhanced (using flag)
python backend/scripts/train.py \
    --saliency-enhanced \
    --tuning backend/checkpoints/rtdetr_r101vd_6x_coco.pth
```

### Testing/Evaluation

```bash
# Original D2-City
python backend/scripts/test.py \
    --config backend/configs/d2city_rtdetr.yml \
    --resume backend/output/d2city_rtdetr_r50vd/checkpoint.pth

# Saliency-Enhanced (using flag)
python backend/scripts/test.py \
    --saliency-enhanced \
    --resume backend/output/d2city_saliency_enhanced_rtdetr_r50vd/checkpoint.pth
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

## Web API

### Backend API Endpoints

The FastAPI backend provides the following endpoints:

#### `POST /upload`

Upload video for processing.

**Request:**

- `file`: Video file (multipart/form-data)
- `pipeline_type` (optional): Pipeline type - `"original"` or `"saliency"` (default: `"original"`)
- `checkpoint_path` (optional): Path to model checkpoint (auto-detected based on pipeline_type)
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

**Query Parameters:**

- `attachment` (optional, default `false`): when `true`, forces `"Content-Disposition: attachment"` so the browser downloads the file immediately.

**Behavior:**

- Supports HTTP `Range` headers (partial content) for inline playback and scrubbing.
- Always returns an H.264 MP4 so modern browsers can decode it without plugins.
- `Accept-Ranges: bytes` is set on full responses; partial requests return `206`.

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
- **Pipeline Selector**: Dropdown to choose between Original D2-City and Saliency-Enhanced pipelines
- **Video Upload**: Drag & drop or click to select
- **Real-time Progress**: Status updates with progress bar and terminal-style logs
- **Video Preview**: Embedded video player with bounding boxes
- **Download**: Direct download link for processed videos
- **Documentation Tab**: Dynamic README rendering with markdown support (updates automatically)
- **Error Handling**: User-friendly error messages
- **Tab Navigation**: Switch between Inference and Documentation views

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

### Scenario 1: Original D2-City Dataset

Place D2-City MP4 video files in:

```
backend/datasets/d2_city/
  ├── video1.mp4
  ├── video2.mp4
  └── ...
```

The dataset loader will:

- Extract frames from videos
- Apply preprocessing (BGR→RGB, resize, normalization)
- Return frames ready for RT-DETR

### Scenario 2: Saliency-Enhanced D2-City Dataset

Place pre-processed saliency-enhanced image files in:

```
backend/datasets/d2_city_saliency_enhanced/
  ├── train/
  │   ├── frame_00001.jpg  (pre-processed, saliency-enhanced)
  │   ├── frame_00002.jpg
  │   └── ...
  └── val/
      ├── frame_00001.jpg
      └── ...
```

The dataset loader will:

- Load pre-processed images directly
- Apply only RT-DETR transforms (resize, tensor conversion)
- No preprocessing needed (already done)

## Scripts

### Primary Scripts

- **`finetuning.py`**: Primary fine-tuning script (uses RT-DETR solver, supports both pipelines via config)
- **`inference.py`**: Primary inference script (returns bbox, labels, scores, centroids)
- **`process_video.py`**: Process videos with bounding boxes (used by FastAPI backend)

### Reference Scripts

- **`train.py`**: Reference training script (supports both pipelines via `--saliency-enhanced` flag or config)
- **`test.py`**: Reference test script (supports both pipelines via `--saliency-enhanced` flag or config)

All scripts support both pipeline scenarios (Original D2-City and Saliency-Enhanced).

## Model Architecture

The model architecture is established from RT-DETR config files:

- **Backbone**: ResNet/DLA/RegNet (defined in config)
- **Encoder**: Hybrid Encoder
- **Decoder**: Transformer Decoder

Pretrained weights are loaded into this architecture for fine-tuning.

## Outputs

- **Checkpoints**:
  - Original pipeline: `backend/output/d2city_rtdetr_r50vd/`
  - Saliency-enhanced pipeline: `backend/output/d2city_saliency_enhanced_rtdetr_r50vd/`
- **Videos**: Annotated videos with bounding boxes (saved in `backend/outputs/`)
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
- **Styling**: Tailwind CSS with dark theme
- **HTTP Client**: Axios
- **State Management**: React hooks
- **Markdown Rendering**: react-markdown with remark-gfm
- **Icons**: Lucide React

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
- `react-markdown@^9.0.0`
- `remark-gfm@^4.0.0`
- `lucide-react@^0.294.0`
- `tailwindcss@^3.4.18`

## Troubleshooting

### Backend Issues

- **Backend won't start**: Check that checkpoint exists in `backend/checkpoints/` or `backend/output/`
- **Model not loading**: RT-DETR is included in `backend/src/rtdetr/` - no external dependency needed
- **Pipeline not found**: Ensure appropriate checkpoint exists for selected pipeline type
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

## Pipeline Scenarios

The system supports two distinct pipeline scenarios:

### Scenario 1: Original D2-City Dataset

- **Data Loading**: ✅ Loads MP4 videos, extracts frames
- **Preprocessing**: ✅ Built-in (BGR→RGB, resize, normalization)
- **Dataset**: `D2CityDatasetRTDETR`
- **Config**: `d2city_rtdetr.yml`
- **Checkpoint**: `backend/output/d2city_rtdetr_r50vd/checkpoint.pth`

### Scenario 2: Saliency-Enhanced D2-City Dataset

- **Data Loading**: ✅ Loads pre-processed image files
- **Preprocessing**: ❌ Not needed (already done)
- **Dataset**: `SaliencyEnhancedD2CityDatasetRTDETR`
- **Config**: `d2city_saliency_enhanced_rtdetr.yml`
- **Checkpoint**: `backend/output/d2city_saliency_enhanced_rtdetr_r50vd/checkpoint.pth`

Both scenarios are fully supported in training, inference, and web interface.

## Next Steps

1. **Fine-tune on Saliency Dataset**: Once saliency-based D2-City dataset is ready
2. **Deploy**: Consider Docker containers for production
3. **Optimize**: Add video compression, caching, CDN
4. **Enhance**: Add batch processing, user accounts, history

## Notes

- Backend processes videos in background (non-blocking)
- Model is cached per pipeline type (faster subsequent requests)
- Frontend polls status every 2 seconds
- Videos are stored locally (consider cloud storage for production)
- CORS is configured for localhost:3000
- Pipeline selector in frontend automatically selects appropriate checkpoint and config
- Both pipelines can be trained independently and used via web interface

## License

This project uses RT-DETR, which is licensed under Apache 2.0.
