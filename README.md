
---
title: My Backend App
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Traffic Violation Detection System

A production-ready computer vision system for detecting traffic violations, extracting license plate information, and managing violation evidence. The backend is a FastAPI application with lazy-loaded YOLO models; a Next.js frontend (deployed separately on Vercel) communicates with the API for analysis, violation logs, statistics, and face matching.

## Features

### Core Detection Capabilities
- **Vehicle Detection**: Detects cars, motorcycles, trucks, buses, and pedestrians using YOLOv11 models (BDD100K primary, UVH fallback for 3-wheelers and specific vehicle types)
- **License Plate Recognition**: Detects license plates on cropped vehicle regions and extracts text using RapidOCR
- **Helmet Detection**: Identifies riders without helmets on the full frame
- **Seatbelt Detection**: Detects drivers/passengers not wearing seatbelts on the full frame
- **Triple Riding Detection**: Flags two-wheelers with three or more detected persons inside the vehicle bounding box
- **Stop-Line Video Analysis**: Processes uploaded video for stop-line violations with optional custom stop line and traffic light state
- **Face Matching**: Compares uploaded face images against stored violation images or scans the full violation database using DeepFace (Facenet512 + RetinaFace)

### Data Collection and Retraining Support
- **Persistent Evidence Storage**: Violation JSON and annotated images stored under `/data/evidence` in Docker/Hugging Face Spaces, or `./evidence` locally
- **Collection API**: Enriched violation records with browser-accessible image URLs for frontend logs and map heatmaps
- **RL Self-Verification**: Human review workflow that copies verified samples into `/data/rl_training/confirmed` or `/data/rl_training/corrections` for model retraining

### Architecture
- **Modular Design**: Each processing step is independent and easily extensible
- **Lazy Model Loading**: Models load only when first needed, reducing memory overhead
- **Lenient Error Handling**: Partial results returned even if some detection steps fail
- **Millisecond-Precision Evidence**: Timestamped records prevent data collisions
- **Configurable Everything**: All thresholds, classes, and parameters in `pipeline/config.py`
- **CORS Enabled**: Backend accepts requests from local dev and Vercel deployments

---

## Project Structure

```
traffic-management-system/
├── pipeline/                       # Backend pipeline package
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration and paths
│   ├── models.py                   # Lazy-loading YOLO model manager
│   ├── preprocessing.py            # Image preprocessing chain
│   ├── ocr_processor.py            # License plate OCR (RapidOCR)
│   ├── pipeline.py                 # Main TrafficViolationPipeline class
│   ├── stopline.py                 # Stop-line video violation detection
│   ├── face_preprocessor.py        # DeepFace embedding and comparison
│   ├── face_match_service.py       # Face match API orchestration
│   ├── data_collection.py          # Collection, heatmap, RL verification storage
│   └── utils.py                    # Evidence I/O, GPS validation, annotation helpers
│
├── app/                            # Next.js frontend pages
├── components/                     # React UI components (analyze, violations, stats, face match)
├── lib/api.ts                      # Frontend API client and types
│
├── evidence/                       # Local evidence storage (when /data/evidence is absent)
│   ├── 2025-06-18T14-30-22.123Z.json
│   ├── 2025-06-18T14-30-22.123Z.jpg
│   └── ...
│
├── face_database/                  # Optional named face embeddings (.npy or embeddings.json)
│
├── main.py                         # FastAPI application and route definitions
├── Dockerfile                      # CUDA-enabled container for Hugging Face Spaces
├── requirements.txt                # Python dependencies
├── package.json                    # Next.js frontend dependencies
├── .env.example                    # Frontend/backend environment variables
├── vercel.json                     # Vercel deployment config
│
├── bdd100k_opensource.pt           # Vehicle detection model (primary)
├── UVH-26-MV-YOLOv11-S.pt         # Vehicle detection model (3-wheelers / UVH triggers)
├── License_plate_bb.pt             # License plate detection
├── helmet_detection.pt             # Helmet detection
├── seatbelt_detection.pt           # Seatbelt detection
│
├── scripts/stopline_pipeline_video.py   # Standalone stop-line script
├── facial_recognition.ipynb             # Face verification experiments
└── credits.md                           # Model and library attribution
```

### Persistent Storage (Docker / Hugging Face Spaces)

| Path | Purpose |
|------|---------|
| `/data/evidence` | Violation JSON, annotated images, and stop-line video outputs |
| `/data/rl_training/confirmed` | Human-verified violations with correct model outputs |
| `/data/rl_training/corrections` | False positives or model errors flagged for retraining |

Locally, evidence defaults to `./evidence` and RL data to `./rl_training` unless overridden by environment variables.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+ (for the Next.js frontend)
- CUDA-capable GPU (optional; CPU works but slower)
- 8GB+ RAM recommended

### 1. Backend Setup

```bash
# Navigate to project directory
cd traffic-management-system

# Create and activate virtual environment (Windows)
python -m venv traffic_venv
traffic_venv\Scripts\activate

# Create and activate virtual environment (Linux/Mac)
python -m venv traffic_venv
source traffic_venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
# Default port 7860 (matches Hugging Face Spaces)
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 7860
```

Server runs at: `http://localhost:7860`

### 3. Frontend Setup (Optional)

```bash
# Copy environment template and set backend URL
cp .env.example .env.local

# Install and run Next.js dev server
npm install
npm run dev
```

Frontend runs at: `http://localhost:3000` and calls the backend via `NEXT_PUBLIC_API_BASE`.

---

## API Endpoints

### Static File Serving

| Path | Description |
|------|-------------|
| `GET /evidence/{filename}` | Serves evidence images, JSON, and video files |
| `GET /view-evidence/{filename}` | Alias for persistent `/data/evidence` (mounted when directory exists) |
| `GET /view-training/{category}/{filename}` | Serves RL training images and JSON from `/data/rl_training` |

---

### 1. POST /analyze - Image Analysis

Analyze an image for traffic violations.

**Request:**
```bash
curl -X POST "http://localhost:7860/analyze" \
  -F "file=@image.jpg" \
  -F "timestamp=2025-06-18T14:30:22.123Z" \
  -F "gps_lat=28.6139" \
  -F "gps_lon=77.2090"
```

**Parameters:**
- `file` (required): Image file (JPEG, PNG, etc.)
- `timestamp` (optional): ISO format timestamp. Default: current UTC time
- `gps_lat` (optional): Latitude coordinate. Default: 0.0
- `gps_lon` (optional): Longitude coordinate. Default: 0.0

**Response:**
```json
{
  "success": true,
  "timestamp": "2025-06-18T14:30:22.123Z",
  "gps": {"latitude": 28.6139, "longitude": 77.2090},
  "violations_count": 2,
  "plates_count": 1,
  "evidence": {
    "timestamp": "2025-06-18T14:30:22.123Z",
    "gps": [28.6139, 77.2090],
    "plates": [
      {
        "text": "DL1AB1234",
        "confidence": 0.92,
        "box": [100, 50, 150, 80]
      }
    ],
    "violations": [
      {
        "type": "no_helmet",
        "vehicle_class": "unknown",
        "confidence": 0.87,
        "box": [200, 100, 250, 180]
      }
    ],
    "annotated_image_path": "/path/to/evidence/2025-06-18T14-30-22.123Z.jpg"
  },
  "annotated_image_base64": "iVBORw0KGgo..."
}
```

---

### 2. POST /analyze-video - Stop-Line Video Analysis

Process an uploaded video for stop-line violations.

**Parameters:**
- `file` (required): Video file (mp4, mov)
- `stop_line` (optional): Normalized stop line as JSON `[x1,y1,x2,y2]` (0..1)
- `initial_light_state` (optional): `"red"` or `"green"`. Default: `"red"`
- `conf_thres` (optional): Detection confidence threshold. Default: 0.3

**Response includes:** `annotated_video_url`, `results_json_url`, `violations_count`, and per-frame violation details.

---

### 3. GET /violations - Retrieve All Evidence Records

Returns all stored evidence JSON files (image records and stop-line records).

```bash
curl "http://localhost:7860/violations"
```

---

### 4. GET /stats - Aggregated Statistics

Returns violation counts, plate counts, and per-type breakdowns across all evidence records.

```bash
curl "http://localhost:7860/stats"
```

---

### 5. POST /face-match/compare - Compare Two Face Images

Compare an uploaded reference person image against a second uploaded image.

**Parameters:** `person_image`, `target_image` (multipart file uploads)

---

### 6. POST /face-match/violation - Match Against Stored Violations

Compare a person image against one stored violation or scan the entire violation database.

**Parameters:**
- `person_image` (required)
- `violation_id` (optional): Evidence timestamp stem to match against
- `scan_all_violations` (optional): Set to `true` to scan all stored violation images

Provide either `violation_id` or `scan_all_violations=true`, not both.

Similarity threshold is controlled by `FACE_SIMILARITY_THRESHOLD` (default 0.50).

---

### 7. Data Collection Endpoints

These endpoints provide enriched data for frontend violation logs, map heatmaps, and RL retraining workflows.

#### GET /collection/violations

Full violation log with timestamps, GPS, plates, violations, and browser-accessible URLs.

Query parameter: `include_zero_gps` (default `true`)

Each record includes: `id`, `gps_lat`, `gps_lon`, `image_url`, `json_url`, and `system_outputs`.

#### GET /collection/violations/{violation_id}

Single violation record with system model output flags (OCR, license plate, vehicle detection).

#### GET /collection/heatmap

GPS points formatted for map heatmap rendering.

Query parameter: `include_zero_gps` (default `false`)

Each point includes: `lat`, `lon`, `timestamp`, `violation_types`, `weight`, `image_url`.

#### POST /collection/verify

Submit human verification for RL retraining. Copies the violation image and labels into the RL training folder. Original evidence remains in place.

**Request body:**
```json
{
  "violation_id": "2025-06-18T14-30-22.123Z",
  "violation_confirmed": true,
  "ocr": { "detected": true, "correct": false, "notes": "Wrong plate text" },
  "license_plate": { "detected": true, "correct": true },
  "vehicle": { "detected": true, "correct": true },
  "helmet": { "detected": true, "correct": true },
  "seatbelt": { "detected": false, "correct": true },
  "annotation_notes": "Optional overall note"
}
```

Routing logic:
- **confirmed/**: Violation is real and all model labels marked correct
- **corrections/**: False positive or any model marked `correct: false`

#### GET /collection/training

List RL training samples. Optional query parameter: `category=confirmed` or `category=corrections`.

#### GET /collection/training/stats

Returns dataset counts, per-model error counts, and `ready_for_retraining` when corrections reach the configured threshold (default 1000, set via `RL_RETRAIN_THRESHOLD`).

---

### 8. GET /health - Health Check

Returns service status, evidence folder path, RL training folder path, and mount status.

---

### 9. GET /info - API Information

Returns API metadata and a catalog of available endpoints.

---

## Configuration

All configuration is centralized in `pipeline/config.py`. Key settings:

```python
# Model paths (project root .pt files)
MODEL_PATHS = {
    "bdd100k": "bdd100k_opensource.pt",
    "uvh": "UVH-26-MV-YOLOv11-S.pt",
    "license_plate": "License_plate_bb.pt",
    "helmet": "helmet_detection.pt",
    "seatbelt": "seatbelt_detection.pt",
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "bdd100k": 0.45,
    "uvh": 0.45,
    "license_plate": 0.4,
    "helmet": 0.4,
    "seatbelt": 0.4,
}

# Violation class filters (model output class names)
VIOLATION_CLASSES = {
    "helmet": "Without Helmet",
    "seatbelt": "person-noseatbelt",
}
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` / `API_PORT` | `7860` | API server port |
| `EVIDENCE_FOLDER` | `/data/evidence` or `./evidence` | Evidence storage path |
| `RL_TRAINING_FOLDER` | `/data/rl_training` or `./rl_training` | RL training dataset path |
| `RL_RETRAIN_THRESHOLD` | `1000` | Corrections count before retraining readiness |
| `FACE_SIMILARITY_THRESHOLD` | `0.50` | Face match cosine similarity threshold |
| `DEEPFACE_HOME` | Project root | DeepFace model weights location |
| `CORS_ORIGINS` | `localhost:3000` | Allowed frontend origins |
| `CORS_ORIGIN_REGEX` | `https://.*\.vercel\.app` | Regex for Vercel deployments |
| `NEXT_PUBLIC_API_BASE` | HF Space URL | Frontend backend URL (`.env.local`) |

---

## System Architecture

### Image Pipeline Processing Flow

```
User Image
    |
[1] Input Validation (GPS, Timestamp)
    |
[2] Image Preprocessing
    |-- Undistort (lens correction)
    |-- Enhance lighting (CLAHE)
    |-- Reduce motion blur (unsharp mask)
    |-- ROI mask (top 20% masked to focus on road)
    |
[3] Vehicle Detection (BDD100K; UVH if 3-wheeler or UVH trigger vehicles detected)
    |
[4] License Plate Detection (on cropped vehicle regions)
    |
[5] OCR - Extract Plate Text (RapidOCR with white border padding)
    |
[6] Violation Detection (global, full frame)
    |-- Helmet detection (Without Helmet class)
    |-- Seatbelt detection (person-noseatbelt class)
    |-- Triple riding (3+ persons inside two-wheeler bounding box)
    |
[7] Annotate Image (green vehicles, yellow plates, red violations)
    |
[8] Generate Evidence (timestamped JSON + annotated JPG)
    |
API Response (JSON + base64 annotated image)
```

### Stop-Line Video Pipeline

```
Uploaded Video
    |
Vehicle detection + multi-object tracking per frame
    |
Stop-line crossing detection (segment intersection)
    |
Traffic light state gating (red light violations)
    |
Annotated video + JSON results saved to evidence folder
```

---

## Class Mappings

### Vehicle Classes (BDD100K)
```python
{
    "car": 0,
    "bike": 1,
    "truck": 2,
    "pedestrian": 3,
    "rider": 4,
    "bus": 5,
    "train": 6
}
```

### Violation Types
- `no_helmet`: Rider without helmet
- `no_seatbelt`: Driver/passenger without seatbelt
- `triple_riding`: Three or more persons on a two-wheeler

Stop-line video violations use a separate schema with `track_id`, `frame_idx`, `bbox`, and `light_state` fields.

### Violation Classes (from detection models)
- **Helmet Model**: Filters for `Without Helmet`
- **Seatbelt Model**: Filters for `person-noseatbelt`

---

## How to Add New Violations

Adding new violations requires three changes:

### Step 1: Add to Config (`pipeline/config.py`)
```python
MODEL_PATHS = {
    # ... existing ...
    "new_violation": "new_violation_model.pt",
}

VIOLATION_CLASSES = {
    # ... existing ...
    "new_violation": "target_class_name",
}
```

### Step 2: Add Loader (`pipeline/models.py`)
```python
def load_new_violation_model(self):
    if self.new_violation_model is None:
        self.new_violation_model = YOLO(Config.MODEL_PATHS["new_violation"])
    return self.new_violation_model
```

### Step 3: Add Detection Logic (`pipeline/pipeline.py`)
```python
def detect_new_violation(self, image):
    model = self.model_manager.load_new_violation_model()
    # ... detection code ...
    return violations
```

No changes to the FastAPI route structure are required unless a new input modality (e.g. a new media type) is introduced.

---

## Input Validation

### GPS Validation
- Latitude: -90 to 90
- Longitude: -180 to 180

### Timestamp Validation
- Format: ISO 8601 (e.g., `"2025-06-18T14:30:22.123Z"`)
- Millisecond precision to prevent filename collisions

### Image Validation
- Must be decodable JPEG/PNG
- Must contain valid image data

---

## Performance Notes

### Model Performance (approximate)
- **Vehicle Detection**: ~150-200ms per image (GPU) / ~500-800ms (CPU)
- **License Plate Detection**: ~50-100ms per vehicle (GPU) / ~200-300ms (CPU)
- **OCR**: ~100-150ms per plate
- **Helmet/Seatbelt Detection**: ~50-100ms per image (GPU) / ~200-300ms (CPU)

### Memory Usage
- Single model: ~500MB - 2GB
- All models loaded: ~5-8GB (lazy loading reduces peak usage)
- GPU VRAM: 2-4GB recommended

### Optimization Tips
1. Use GPU with CUDA-compatible PyTorch for 3-5x speedup
2. Increase Uvicorn workers only when sufficient GPU/CPU memory is available
3. Consider quantized models for edge deployment

---

## Testing

### Test Image Analysis
```bash
curl -X POST "http://localhost:7860/analyze" \
  -F "file=@test_image.jpg" \
  -F "timestamp=2025-06-18T14:30:22.123Z" \
  -F "gps_lat=28.6139" \
  -F "gps_lon=77.2090"
```

### Check Evidence Records
```bash
curl "http://localhost:7860/violations" | python -m json.tool
```

### Check Collection Data (with image URLs)
```bash
curl "http://localhost:7860/collection/violations" | python -m json.tool
curl "http://localhost:7860/collection/heatmap" | python -m json.tool
```

### View Statistics
```bash
curl "http://localhost:7860/stats" | python -m json.tool
curl "http://localhost:7860/collection/training/stats" | python -m json.tool
```

---

## Evidence Record Format

### Image Analysis Record

Each image analysis produces a paired JSON and annotated JPG file:

```json
{
  "timestamp": "2025-06-18T14:30:22.123Z",
  "gps": [28.6139, 77.2090],
  "plates": [
    {
      "text": "DL1AB1234",
      "confidence": 0.92,
      "box": [x1, y1, x2, y2]
    }
  ],
  "violations": [
    {
      "type": "no_helmet",
      "vehicle_class": "unknown",
      "confidence": 0.87,
      "box": [x1, y1, x2, y2]
    },
    {
      "type": "triple_riding",
      "vehicle_class": "motorcycle",
      "confidence": 0.75,
      "box": [x1, y1, x2, y2],
      "person_count": 3
    }
  ],
  "annotated_image_path": "/absolute/path/to/2025-06-18T14-30-22.123Z.jpg"
}
```

Filenames derive from the timestamp with colons replaced by dashes:
`2025-06-18T14:30:22.123Z` becomes `2025-06-18T14-30-22.123Z`.

### RL Training Record

Verified samples stored under `/data/rl_training/` include the original evidence snapshot, human labels per model (OCR, license plate, vehicle, helmet, seatbelt), category, and browser-accessible image URLs.

---

## Known Limitations

1. **3-Wheeler Detection**: Uses UVH model when triggered; accuracy depends on model quality
2. **OCR Accuracy**: RapidOCR works best with clear, frontal plate images
3. **Helmet/Seatbelt Detection**: Runs globally on the full frame; not linked to specific vehicle boxes
4. **Triple Riding**: Uses helmet-model detections as person proxies inside two-wheeler boxes
5. **Mixed Evidence Schemas**: Stop-line video JSON records appear alongside image records in `/violations`
6. **Face Matching**: Requires DeepFace weights; optional face database must be populated manually
7. **GPU Memory**: All models cannot run simultaneously on less than 4GB VRAM

---

## Extending the System

### Adding a New Model
1. Add model path to `config.py`
2. Add loader method to `ModelManager` in `models.py`
3. Create detection method in `TrafficViolationPipeline`
4. Optionally add a new API endpoint in `main.py`

### Custom Preprocessing
Edit `preprocessing.py`:
- Add a new function (e.g., contrast adjustment)
- Call it in the `preprocess_frame()` chain

### Custom Annotation Logic
Edit `draw_bounding_boxes()` in `utils.py` to customize colors, labels, or overlays.

---

## Troubleshooting

### "Model not found" Error
- Verify `.pt` model files exist in the project root
- Check paths in `pipeline/config.py`

### Out of Memory Error
- Reduce image size before sending
- Use CPU inference or reduce concurrent workers
- Ensure lazy loading is not pre-loading all models

### Low Detection Accuracy
- Adjust confidence thresholds in `config.py`
- Verify image quality (lighting, resolution, angle)
- Confirm models are appropriate for your scene type

### OCR Not Extracting Text
- Check plate crop quality
- Ensure RapidOCR is installed: `pip install rapidocr-onnxruntime`
- Adjust `white_border_px` in `config.py` OCR parameters

### Evidence Not Persisting on Hugging Face Spaces
- Confirm the Space persistent volume is mounted at `/data`
- Verify `/data/evidence` exists inside the container
- Override with `EVIDENCE_FOLDER=/data/evidence` if needed

---

## Dependencies

### Python (requirements.txt)

| Package | Purpose |
|---------|---------|
| fastapi | Web framework |
| uvicorn | ASGI server |
| python-multipart | File upload support |
| ultralytics | YOLO model inference |
| opencv-python-headless | Image processing |
| numpy | Numerical computing (< 2.0) |
| rapidocr-onnxruntime | License plate OCR |
| pillow | Image handling |
| deepface | Face embedding and comparison |
| tf-keras | DeepFace backend dependency |

PyTorch is provided by the Docker base image (`pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`).

### Frontend (package.json)

Next.js 16, React 19, Tailwind CSS 4, SWR for data fetching, shadcn/ui components.

---

## Deployment

### Docker / Hugging Face Spaces

The included `Dockerfile` builds a CUDA-enabled container exposed on port 7860. Persistent directories are created at build time:

```
/data/evidence
/data/rl_training/confirmed
/data/rl_training/corrections
```

The backend is deployed to Hugging Face Spaces (`Bikki26/traffic-management-system`) via GitHub Actions sync (`.github/workflows/sync_to_hf.yml`).

### Local Docker Run

```bash
docker build -t traffic-violation-api .
docker run -p 7860:7860 \
  -v $(pwd)/evidence:/data/evidence \
  -v $(pwd)/rl_training:/data/rl_training \
  traffic-violation-api
```

### Frontend (Vercel)

The Next.js frontend deploys to Vercel with `NEXT_PUBLIC_API_BASE` pointing at the Hugging Face Space URL. See `.env.example` for required variables.

---

## License

This project uses pre-trained models. Ensure compliance with each model's license before production deployment. See `credits.md` for attribution.

---

## Developer Notes

### Code Style
- Follow PEP 8
- Add docstrings to functions
- Use type hints where practical
- Keep functions focused and modular

### Logging
- Errors are printed to console via `print()` in route handlers
- Replace with the `logging` module for production hardening

### Testing Checklist
- Test with images of varying quality and lighting
- Test with different vehicle types (2-wheelers, cars, trucks)
- Validate GPS and timestamp boundary conditions
- Verify evidence files are written and served correctly
- Test collection and heatmap endpoints with GPS-enabled records

---

**Last Updated**: June 22, 2026  
**System Version**: 1.0.0  
**Status**: Production Ready
