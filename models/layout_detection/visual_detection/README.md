# DocLayout-YOLO Integration

This module integrates [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for document layout analysis, providing state-of-the-art document element detection capabilities.

## Overview

DocLayout-YOLO is a real-time and robust layout detection model that can identify various document elements including:

- **Text blocks**
- **Titles** 
- **Figures** and **Figure Captions**
- **Tables** and **Table Captions**
- **Headers** and **Footers**
- **References**
- **Equations**

## Features

- **Multiple Model Support**: Choose from different pre-trained models (DocStructBench, D4LA, DocLayNet)
- **Automatic Model Download**: Models are automatically downloaded from Hugging Face Hub
- **Batch Processing**: Process multiple images or PDF pages at once
- **Visualization**: Generate annotated images with bounding boxes and labels
- **API Integration**: REST API endpoints for web service integration
- **Flexible Input**: Support for images (PNG, JPG, etc.) and PDF documents

## Installation

The required dependencies are already included in the main `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `doclayout-yolo`
- `huggingface-hub>=0.16.0`
- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `opencv-python>=4.5.0`
- `ultralytics>=8.0.0`

## Quick Start

### 1. Basic Usage

```python
from models.layout_detection import DocLayoutDetector

# Initialize detector
detector = DocLayoutDetector(model_name="docstructbench")

# Detect layout elements
result = detector.detect("path/to/your/image.png")

# Get detected elements
elements = result.get_elements()
for element in elements:
    print(f"{element['type']}: confidence={element['confidence']:.2f}")
    print(f"  Location: ({element['bbox']['x1']}, {element['bbox']['y1']}) to ({element['bbox']['x2']}, {element['bbox']['y2']})")
```

### 2. Visualization

```python
# Create annotated image
annotated_img = detector.visualize(
    "path/to/your/image.png", 
    save_path="annotated_output.png"
)
```

### 3. API Usage

Start the API server:

```bash
python -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

#### Detect Layout Elements

```bash
curl -X POST "http://localhost:8000/api/detect-layout" \
     -F "file=@your_document.pdf" \
     -F "confidence=0.25" \
     -F "model_name=docstructbench"
```

#### Get Annotated Images

```bash
curl -X POST "http://localhost:8000/api/visualize-layout" \
     -F "file=@your_document.pdf" \
     -F "confidence=0.25" \
     -F "page_number=0"
```

## Available Models

| Model | Description | Use Case |
|-------|------------|----------|
| `docstructbench` | Fine-tuned on DocStructBench for various document types | **Recommended** - General purpose |
| `d4la` | Trained on D4LA dataset | Academic papers and documents |
| `doclaynet` | Trained on DocLayNet dataset | Published documents and papers |

## Command Line Tools

### Demo Script

```bash
python demo_layout_detection.py --image path/to/image.png --model docstructbench
```

Options:
- `--image`: Input image path (required)
- `--output`: Output annotated image path (optional)
- `--model`: Model to use (`docstructbench`, `d4la`, `doclaynet`)
- `--confidence`: Confidence threshold (0.0-1.0, default: 0.25)
- `--device`: Device (`auto`, `cpu`, `cuda`, `cuda:0`, etc.)
- `--image-size`: Input image size (default: 1024)

### Model Management

Download a specific model:
```bash
python -m models.layout_detection.download_model --model docstructbench
```

List available models:
```bash
python -m models.layout_detection.download_model --list
```

### Testing

Run integration tests:
```bash
python test_layout_detection.py
```

## API Endpoints

### POST `/api/detect-layout`

Detect layout elements in an uploaded image or PDF.

**Parameters:**
- `file`: Image file (PNG, JPG, etc.) or PDF file
- `confidence`: Confidence threshold (0.0-1.0, default: 0.25)
- `model_name`: Model to use (default: "docstructbench")
- `image_size`: Input image size (default: 1024)

**Response:**
```json
{
  "filename": "document.pdf",
  "file_type": "application/pdf",
  "model_used": "docstructbench",
  "confidence_threshold": 0.25,
  "total_pages": 3,
  "total_elements": 15,
  "element_type_summary": {
    "Text": 8,
    "Title": 3,
    "Figure": 2,
    "Table": 2
  },
  "pages": [
    {
      "page_number": 0,
      "elements": [
        {
          "id": 0,
          "type": "Title",
          "class_id": 1,
          "confidence": 0.95,
          "bbox": {
            "x1": 100,
            "y1": 50,
            "x2": 500,
            "y2": 80,
            "width": 400,
            "height": 30
          }
        }
      ],
      "total_elements": 5
    }
  ],
  "success": true
}
```

### POST `/api/visualize-layout`

Detect layout and return annotated images.

**Parameters:**
- `file`: Image file or PDF file
- `confidence`: Confidence threshold (default: 0.25)
- `model_name`: Model to use (default: "docstructbench")
- `image_size`: Input image size (default: 1024)
- `page_number`: Page number for PDFs (0-based, -1 for all pages)

**Response:**
```json
{
  "filename": "document.pdf",
  "model_used": "docstructbench",
  "total_pages": 1,
  "annotated_pages": [
    {
      "page_number": 0,
      "annotated_image": "base64_encoded_image_data...",
      "total_elements": 5
    }
  ],
  "success": true
}
```

## Advanced Usage

### Filtering Results

```python
# Filter by confidence
high_conf_result = result.filter_by_confidence(min_confidence=0.8)

# Filter by element types
tables_and_figures = result.filter_by_type(["Table", "Figure"])
```

### Batch Processing

```python
# Process multiple images
images = ["image1.png", "image2.png", "image3.png"]
results = detector.detect_batch(images)

for i, result in enumerate(results):
    print(f"Image {i}: {len(result.get_elements())} elements detected")
```

### Custom Configuration

```python
detector = DocLayoutDetector(
    model_name="docstructbench",
    device="cuda:0",  # Use specific GPU
    confidence_threshold=0.3,
    image_size=1536  # Higher resolution for better accuracy
)
```

## Element Types

The model can detect the following document elements:

| Class ID | Element Type | Description |
|----------|-------------|-------------|
| 0 | Text | Regular text blocks |
| 1 | Title | Document or section titles |
| 2 | Figure | Images, charts, diagrams |
| 3 | Figure Caption | Captions for figures |
| 4 | Table | Data tables |
| 5 | Table Caption | Captions for tables |
| 6 | Header | Page headers |
| 7 | Footer | Page footers |
| 8 | Reference | Bibliography/references |
| 9 | Equation | Mathematical equations |

## Performance Tips

1. **Model Selection**: Use `docstructbench` for general documents, specialized models for specific domains
2. **Image Size**: Larger images (1536px) provide better accuracy but slower processing
3. **Confidence Threshold**: Lower values (0.2-0.3) detect more elements, higher values (0.5+) for higher precision
4. **Device**: Use GPU (`cuda`) for faster processing if available
5. **Batch Processing**: Process multiple images together for better efficiency

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Model Download Fails**: Check internet connection and Hugging Face access
   ```python
   from models.layout_detection import download_model
   download_model("docstructbench", force_download=True)
   ```

3. **CUDA Out of Memory**: Reduce image size or use CPU
   ```python
   detector = DocLayoutDetector(device="cpu", image_size=512)
   ```

4. **No Elements Detected**: Try lowering confidence threshold
   ```python
   result = detector.detect(image, confidence_threshold=0.1)
   ```

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## References

- [DocLayout-YOLO Paper](https://arxiv.org/abs/2410.12628)
- [DocLayout-YOLO GitHub](https://github.com/opendatalab/DocLayout-YOLO)
- [DocLayout-YOLO Hugging Face](https://huggingface.co/spaces/opendatalab/DocLayout-YOLO)

## Citation

If you use this integration in your research, please cite:

```bibtex
@misc{zhao2024doclayoutyoloenhancingdocumentlayout,
      title={DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception}, 
      author={Zhiyuan Zhao and Hengrui Kang and Bin Wang and Conghui He},
      year={2024},
      eprint={2410.12628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.12628}, 
}
``` 