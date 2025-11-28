# Implementation Branches

## Overview

Two separate implementation branches handle different data sources: original D2-City videos and pre-processed saliency-enhanced frames.

## Branch 1: Original D2-City Dataset

### Dataset Class
**`D2CityDatasetRTDETR`** (`src/datasets/d2city_dataset_rtdetr.py`)

### Data Source
- **Format**: MP4 video files
- **Location**: `./data/d2_city/` (or specified `root_dir`)
- **Structure**: Directory containing `.mp4` files

### Data Loading Process

1. **Video Frame Extraction**
   ```python
   # Opens video with OpenCV
   cap = cv2.VideoCapture(video_path)
   cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
   ret, frame = cap.read()
   ```

2. **Preprocessing**
   - BGR → RGB conversion (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`)
   - PIL Image conversion
   - RT-DETR transforms applied (resize, normalization, augmentation)

3. **Frame Sampling**
   - `frame_skip`: Extract every Nth frame (default: 5)
   - `max_frames_per_video`: Limit frames per video

### Configuration
**File**: `configs/d2city_rtdetr.yml`

```yaml
train_dataloader:
  dataset:
    type: D2CityDatasetRTDETR
    root_dir: ./datasets/d2_city
    frame_skip: 5
    max_frames_per_video: 100
```

### Characteristics
- ✅ Extracts frames from videos on-the-fly
- ✅ Applies basic preprocessing (BGR→RGB)
- ✅ No annotations (empty target dicts)
- ⚠️ Requires video decoding (slower)

---

## Branch 2: Saliency-Enhanced Dataset

### Dataset Class
**`SaliencyEnhancedD2CityDatasetRTDETR`** (`src/datasets/saliency_enhanced_d2city_dataset_rtdetr.py`)

### Data Source
- **Format**: Pre-processed image files (`.jpg`, `.png`)
- **Location**: `./data/T2-saliency/enhanced_frames/`
- **Structure**: 
  ```
  enhanced_frames/
    video_id_1/
      frame_00001.jpg
      frame_00002.jpg
    video_id_2/
      frame_00001.jpg
  ```

### Data Loading Process

1. **Direct Image Loading**
   ```python
   # Loads pre-processed images directly
   img = Image.open(image_path).convert('RGB')
   ```

2. **No Preprocessing**
   - Frames are already:
     - RGB format
     - Saliency-enhanced (Frame × Saliency Mask)
     - Pre-processed
   - Only RT-DETR transforms applied (resize, normalization, augmentation)

3. **Annotation Loading** (if available)
   - XML annotation files in `annotations_dir`
   - Parsed via `d2city_annotation_parser.py`
   - Maps D2-City labels to COCO class indices

### Configuration
**File**: `configs/d2city_saliency_enhanced_rtdetr_r101vd.yml`

```yaml
train_dataloader:
  dataset:
    type: SaliencyEnhancedD2CityDatasetRTDETR
    root_dir: ./data/T2-saliency/enhanced_frames
    annotations_dir: ./data/annotations
```

### Characteristics
- ✅ Loads pre-processed frames directly (faster)
- ✅ Supports annotations (for training)
- ✅ No video decoding overhead
- ✅ Saliency enhancement already applied
- ✅ **Backbone**: Uses the ResNet-101-VD config (`d2city_saliency_enhanced_rtdetr_r101vd.yml`) aligned with our COCO checkpoint

---

## Key Differences

| Aspect | Original D2-City | Saliency-Enhanced |
|--------|------------------|-------------------|
| **Input** | MP4 videos | Pre-processed images |
| **Preprocessing** | BGR→RGB conversion | None (already processed) |
| **Data Loading** | Video frame extraction | Direct image loading |
| **Annotations** | Not supported | Supported (XML) |
| **Speed** | Slower (video decoding) | Faster (direct image load) |
| **Use Case** | Raw video processing | Transfer learning with annotations |

## Code Locations

- **Original branch**: `src/datasets/d2city_dataset_rtdetr.py`
- **Saliency branch**: `src/datasets/saliency_enhanced_d2city_dataset_rtdetr.py`
- **Configs**: `configs/d2city_rtdetr.yml` vs `configs/d2city_saliency_enhanced_rtdetr_r101vd.yml`
- **Registration**: `src/datasets/register_rtdetr.py`

## Selection

The branch is selected via the config file's `dataset.type` field:
- `D2CityDatasetRTDETR` → Original branch
- `SaliencyEnhancedD2CityDatasetRTDETR` → Saliency-enhanced branch

## Inference Integration

The FastAPI backend automatically mirrors the frontend pipeline selector:

1. `/upload` accepts `pipeline_type` (`"original"` or `"saliency"`).
2. The backend searches `backend/output/` for the matching checkpoint directory (`d2city_rtdetr_*` vs `d2city_saliency_enhanced_*`) and falls back to `backend/checkpoints/`.
3. The same field determines which YAML config is passed into `YAMLConfig` before establishing the model.
4. During processing, `scripts/process_video.py` re-encodes every output to H.264 so both branches produce browser-friendly previews.
5. **Prediction Logging**: For every completed video processing job, detailed JSON logs are automatically generated containing all detection results (bounding boxes, class labels, confidence scores, timestamps, and statistics). Logs are saved as `{job_id}_output_predictions.json` in `backend/outputs/` and can be retrieved via the `GET /logs/{job_id}` API endpoint.

This guarantees that choosing "Saliency-Enhanced" in the UI always uses the saliency dataset class, the saliency-specific config (`d2city_saliency_enhanced_rtdetr_r101vd.yml`), and the fine-tuned weights saved in `backend/output/d2city_saliency_enhanced_rtdetr_r101vd/`.

