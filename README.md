
---
title: My Backend App
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Traffic Violation Detection System

A production-ready, scalable FastAPI-based computer vision system for detecting traffic violations and extracting license plate information using pre-trained deep learning models.

## 🎯 Features

### Core Detection Capabilities
- **Vehicle Detection**: Detects cars, motorcycles, trucks, buses, and pedestrians using YOLOv11 models
- **License Plate Recognition**: Detects license plates and extracts text using RapidOCR
- **Helmet Detection**: Identifies riders without helmets (2-wheelers)
- **Seatbelt Detection**: Detects drivers/passengers not wearing seatbelts (cars, trucks)
- **Facial Recognition** (Optional): Matches detected faces against a face database using DeepFace

### Architecture
- **Modular Design**: Each processing step is independent and easily extensible
- **Lazy Model Loading**: Models load only when first needed, reducing memory overhead
- **Lenient Error Handling**: Partial results returned even if some detection steps fail
- **Millisecond-Precision Evidence**: Timestamped records prevent data collisions
- **Configurable Everything**: All thresholds, classes, and parameters in one config file

---

## 📁 Project Structure

```
Round2_flipkart_hackathon/
├── pipeline/                    # Main pipeline package
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Centralized configuration
│   ├── models.py               # Lazy-loading model manager
│   ├── preprocessing.py        # Image preprocessing functions
│   ├── ocr_processor.py        # License plate OCR
│   ├── face_processor.py       # Optional facial recognition
│   ├── utils.py                # Utility functions
│   └── pipeline.py             # Main TrafficViolationPipeline class
│
├── evidence/                    # Auto-created: Stored evidence records
│   ├── 2025-06-18T14-30-22.123Z.json
│   ├── 2025-06-18T14-30-22.123Z.jpg
│   └── ...
│
├── main.py                      # FastAPI application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── bdd100k_opensource.pt        # Vehicle detection model (primary)
├── UVH-26-MV-YOLOv11-S.pt      # Vehicle detection model (3-wheelers)
├── License_plate_bb.pt         # License plate detection
├── helmet_detection.pt         # Helmet detection
├── seatbelt_detection.pt       # Seatbelt detection
│
└── face_database/              # Optional: Face embeddings for recognition
    ├── embeddings.json         # Face embeddings dictionary
    └── (or .npy files per person)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, CPU works too)
- 8GB+ RAM recommended

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd Round2_flipkart_hackathon

# Create virtual environment
python -m venv traffic_venv

# Activate (Windows)
traffic_venv\Scripts\activate

# Activate (Linux/Mac)
source traffic_venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the API Server

```bash
# Development (reload on code changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production (no reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server runs at: `http://localhost:8000`

---

## 📡 API Endpoints

### 1. **POST /analyze** - Main Analysis Endpoint

Analyze an image for traffic violations.

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze" \
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
        "vehicle_class": "motorcycle",
        "confidence": 0.87,
        "box": [200, 100, 250, 180]
      }
    ],
    "annotated_image_path": "evidence/2025-06-18T14-30-22.123Z.jpg"
  },
  "annotated_image_base64": "iVBORw0KGgo..."
}
```

---

### 2. **GET /violations** - Retrieve All Violation Records

Get all stored evidence records (most recent first).

**Request:**
```bash
curl "http://localhost:8000/violations"
```

**Response:**
```json
{
  "success": true,
  "total_records": 42,
  "violations": [
    {
      "timestamp": "2025-06-18T15:45:12.456Z",
      "gps": [28.6150, 77.2100],
      "plates": [...],
      "violations": [...],
      "annotated_image_path": "..."
    }
  ]
}
```

---

### 3. **GET /stats** - Get Aggregated Statistics

Get violation counts and statistics across all records.

**Request:**
```bash
curl "http://localhost:8000/stats"
```

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_records": 42,
    "total_violations": 87,
    "total_plates": 41,
    "average_violations_per_record": 2.07,
    "violation_counts": {
      "no_helmet": 45,
      "no_seatbelt": 42
    },
    "vehicle_class_counts": {
      "motorcycle": 45,
      "car": 42
    }
  }
}
```

---

### 4. **POST /recognize-faces** - Facial Recognition (Optional)

Detect faces and match against face database.

**Request:**
```bash
curl -X POST "http://localhost:8000/recognize-faces" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "success": true,
  "faces_detected": 2,
  "faces": [
    {
      "box": [100, 50, 200, 250],
      "identity": "John Doe",
      "confidence": 0.92,
      "distance": 0.35
    }
  ],
  "annotated_image_base64": "iVBORw0KGgo..."
}
```

---

### 5. **GET /health** - Health Check

```bash
curl "http://localhost:8000/health"
```

---

### 6. **GET /info** - API Information

```bash
curl "http://localhost:8000/info"
```

---

## 🔧 Configuration

All configuration is in `pipeline/config.py`:

```python
# Model paths
MODEL_PATHS = {
    "bdd100k": "bdd100k_opensource.pt",
    "uvh": "UVH-26-MV-YOLOv11-S.pt",
    "license_plate": "License_plate_bb.pt",
    "helmet": "helmet_detection.pt",
    "seatbelt": "seatbelt_detection.pt",
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "bdd100k": 0.15,
    "license_plate": 0.5,
    "helmet": 0.5,
    "seatbelt": 0.5,
}

# Violation class filters
VIOLATION_CLASSES = {
    "helmet": "no-helmet",           # Filter for this class only
    "seatbelt": "person-noseatbelt"  # Filter for this class only
}
```

---

## 🏗️ System Architecture

### Pipeline Processing Flow

```
User Image
    ↓
[1] Input Validation (GPS, Timestamp)
    ↓
[2] Image Preprocessing (undistort, enhance, blur-reduce, dehaze)
    ↓
[3] Vehicle Detection (BDD100K or UVH based on 3-wheeler check)
    ↓
[4] License Plate Detection (on cropped vehicle regions for accuracy)
    ↓
[5] OCR - Extract Plate Text (RapidOCR with white border padding)
    ↓
[6] Violation Detection
    ├─ Helmet detection (if motorcycle/2-wheeler)
    └─ Seatbelt detection (if car/truck)
    ↓
[7] Annotate Image (green vehicles, yellow plates, red violations)
    ↓
[8] Generate Evidence (JSON + save timestamped files)
    ↓
API Response (JSON + base64 annotated image)
```

---

## 📊 Class Mappings

### Vehicle Classes (BDD100K)
```python
{
    "car": 0,
    "motorcycle": 1,
    "truck": 2,
    "pedestrian": 3,
    "rider": 4,
    "bus": 5,
    "train": 6,
    "bicycle": 7,
}
```

### Violation Types
- `no_helmet`: Rider without helmet (motorcycles/2-wheelers)
- `no_seatbelt`: Driver/passenger without seatbelt (cars/trucks)

### Violation Classes (from models)
- **Helmet Model**: `['helmet', 'no-helmet']` → We flag `'no-helmet'`
- **Seatbelt Model**: `['person-noseatbelt', 'person-seatbelt', 'seatbelt']` → We flag `'person-noseatbelt'`

---

## 🔄 How to Add New Violations

Adding new violations is simple and requires only 3 changes:

### Step 1: Add to Config (`pipeline/config.py`)
```python
MODEL_PATHS = {
    # ... existing ...
    "stop_line": "stop_line_model.pt",  # NEW
}

VIOLATION_CLASSES = {
    # ... existing ...
    "stop_line": "crossed_line",  # NEW
}
```

### Step 2: Add Loader (`pipeline/models.py`)
```python
class ModelManager:
    def load_stop_line_model(self):  # NEW
        if self.stop_line_model is None:
            self.stop_line_model = YOLO(Config.MODEL_PATHS["stop_line"])
        return self.stop_line_model
```

### Step 3: Add Detection Logic (`pipeline/pipeline.py`)
```python
def detect_stop_line(self, image):
    """Your detection logic here"""
    model = self.model_manager.load_stop_line_model()
    # ... detection code ...
    return violations
```

That's it! No changes needed to FastAPI or core pipeline structure.

---

## 🔐 Input Validation

The system validates all inputs:

### GPS Validation
- Latitude: -90 to 90
- Longitude: -180 to 180

### Timestamp Validation
- Format: ISO 8601 (e.g., "2025-06-18T14:30:22.123Z")
- Millisecond precision to prevent collisions

### Image Validation
- Must be decodable JPEG/PNG
- Must contain valid image data

---

## 📈 Performance Notes

### Model Performance
- **Vehicle Detection**: ~150-200ms per image (GPU) / ~500-800ms (CPU)
- **License Plate Detection**: ~50-100ms per vehicle (GPU) / ~200-300ms (CPU)
- **OCR**: ~100-150ms per plate (CPU/GPU)
- **Helmet/Seatbelt Detection**: ~50-100ms per vehicle (GPU) / ~200-300ms (CPU)

### Memory Usage
- Single model: ~500MB - 2GB
- All models loaded: ~5-8GB (lazy loading reduces this)
- GPU VRAM: 2-4GB recommended

### Optimization Tips
1. **Use GPU**: Install CUDA-compatible PyTorch for 3-5x speedup
2. **Batch Processing**: Submit multiple images in sequence for better throughput
3. **Model Pruning**: Consider quantized models for edge deployment
4. **Workers**: Increase Uvicorn workers for concurrent requests

---

## 🧪 Testing

### Test with Sample Image
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@test_images_for_preprocessing/2.jpg" \
  -F "timestamp=2025-06-18T14:30:22.123Z" \
  -F "gps_lat=28.6139" \
  -F "gps_lon=77.2090"
```

### Check Evidence Records
```bash
curl "http://localhost:8000/violations" | python -m json.tool
```

### View Statistics
```bash
curl "http://localhost:8000/stats" | python -m json.tool
```

---

## 📋 Known Limitations

1. **3-Wheeler Detection**: Uses UVH model; accuracy depends on model quality
2. **OCR Accuracy**: RapidOCR works best with clear, frontal plate images
3. **Helmet Detection**: Works on cropped regions; requires clear headwear visibility
4. **Facial Recognition**: Requires pre-populated face database; not included
5. **GPU Memory**: All models cannot run simultaneously on <4GB VRAM

---

## 🤝 Extending the System

### Adding a New Model
1. Add model path to `config.py`
2. Add loader method to `ModelManager`
3. Create detection method in `TrafficViolationPipeline`
4. (Optional) Add new API endpoint in `main.py`

### Custom Preprocessing
Edit `preprocessing.py`:
- Add new function (e.g., `apply_contrast()`)
- Call in `preprocess_frame()` chain

### Custom Annotation Logic
Edit `draw_bounding_boxes()` in `utils.py` to customize colors, labels, or overlay.

---

## 📝 Evidence Record Format

Each evidence record is stored as JSON:

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
      "vehicle_class": "motorcycle",
      "confidence": 0.87,
      "box": [x1, y1, x2, y2]
    }
  ],
  "annotated_image_path": "evidence/2025-06-18T14-30-22.123Z.jpg"
}
```

---

## 🐛 Troubleshooting

### "Model not found" Error
- Check model file paths in `config.py`
- Ensure `.pt` files are in project root

### Out of Memory Error
- Reduce image size before sending
- Disable non-essential models
- Use CPU-optimized model versions
- Reduce worker count

### Low Detection Accuracy
- Increase confidence threshold (lower = more detections)
- Check image quality (clear, well-lit, frontal)
- Ensure model is trained on similar dataset

### OCR Not Extracting Text
- Check plate image quality
- Ensure RapidOCR is installed: `pip install rapidocr-onnxruntime`
- Try adjusting white border padding in `config.py`

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | 8.0.177 | YOLO model inference |
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| opencv-python | 4.8.1.78 | Image processing |
| rapidocr-onnxruntime | 1.3.21 | License plate OCR |
| deepface | 0.0.100 | Facial recognition |
| torch | 2.0.1 | Deep learning framework |
| numpy | 1.24.3 | Numerical computing |
| pillow | 10.0.1 | Image handling |

---

## 📄 License

This project uses pre-trained models. Ensure compliance with each model's license before production deployment.

---

## 👨‍💻 Developer Notes

### Code Style
- Follow PEP 8
- Add docstrings to all functions
- Use type hints where possible
- Keep functions focused and modular

### Logging
- Use `print()` for now; replace with `logging` module for production
- Check `main.py` console output for error details

### Testing
- Test with images of varying quality and lighting
- Test with different vehicle types
- Validate GPS/timestamp boundaries

---

## 🚢 Deployment

### Docker (Coming Soon)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms
- **AWS**: Elastic Container Service (ECS) or Lambda
- **Google Cloud**: Cloud Run or App Engine
- **Azure**: Container Instances or App Service

---

## 📞 Support

For issues or feature requests, please refer to the error messages in the API response or console output.

---

**Last Updated**: June 18, 2025  
**System Version**: 1.0.0  
**Status**: Production Ready
