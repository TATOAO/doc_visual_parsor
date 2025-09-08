# Lightweight Document Layout Detection

This package provides a clean, lightweight implementation for document layout detection using ONNX models and merging with PyMuPDF content extraction.

## Features

- **ONNX-based Layout Detection**: Uses ONNX Runtime for fast, lightweight layout detection
- **PDF Content Enrichment**: Combines CV detection with PyMuPDF content extraction
- **Clean API**: Simple, intuitive interface for document processing
- **Minimal Dependencies**: No PyTorch, no LLM dependencies, no fuzzy matching

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic ONNX Layout Detection

```python
from doc_chunking.src import ONNXLayoutDetector

# Initialize detector
detector = ONNXLayoutDetector(
    model_path="path/to/model.onnx",
    confidence_threshold=0.25
)

# Detect layout in PDF
result = detector.detect_layout("document.pdf")
print(f"Detected {len(result.elements)} elements")
```

### Hybrid CV + PDF Extraction

```python
from doc_chunking.src import PdfStyleCVMixLayoutExtractor

# Initialize hybrid extractor
extractor = PdfStyleCVMixLayoutExtractor(
    model_path="path/to/model.onnx",
    cv_confidence_threshold=0.25
)

# Extract layout and content
result = extractor.detect_layout("document.pdf")
print(f"Extracted {len(result.elements)} enriched elements")
```

## API Reference

### ONNXLayoutDetector

Lightweight ONNX-based document layout detector.

**Parameters:**
- `model_path`: Path to ONNX model file
- `confidence_threshold`: Minimum confidence for detections (default: 0.25)
- `image_size`: Input image size for model (default: 1024)
- `pdf_dpi`: DPI for PDF to image conversion (default: 150)
- `device`: Device to use ('auto', 'cpu', 'cuda') (default: 'auto')

**Methods:**
- `detect_layout(input_data)`: Detect layout in PDF or image
- `get_supported_formats()`: Get list of supported file formats

### PdfStyleCVMixLayoutExtractor

Hybrid extractor combining CV detection with PDF content enrichment.

**Parameters:**
- `model_path`: Path to ONNX model file
- `cv_confidence_threshold`: Confidence threshold for CV detection (default: 0.25)
- `cv_image_size`: Input image size for CV model (default: 1024)
- `cv_pdf_dpi`: DPI for PDF to image conversion (default: 150)
- `device`: Device to use (default: 'auto')

**Methods:**
- `detect_layout(input_data)`: Extract layout and content from PDF

### Data Structures

#### LayoutElement
- `id`: Unique identifier
- `element_type`: Type of element (TITLE, PLAIN_TEXT, TABLE, etc.)
- `text`: Text content
- `bbox`: Bounding box coordinates
- `confidence`: Detection confidence score
- `style`: Style information (font, formatting, runs)
- `metadata`: Additional metadata

#### LayoutExtractionResult
- `elements`: List of detected layout elements
- `metadata`: Extraction metadata

## Supported Formats

- **Input**: PDF, JPG, PNG, BMP, TIFF
- **Output**: Structured layout elements with text content and formatting

## Dependencies

- `onnxruntime`: ONNX model inference
- `PyMuPDF`: PDF processing
- `opencv-python`: Image processing
- `pydantic`: Data validation
- `numpy`: Numerical operations
- `Pillow`: Image handling

## Migration from Old Version

The new version removes:
- LLM-based title structure extraction
- Fuzzy matching functionality
- PyTorch dependencies
- Complex processor pipelines

To migrate:
1. Replace `CVLayoutDetector` with `ONNXLayoutDetector`
2. Use `PdfStyleCVMixLayoutExtractor` for hybrid extraction
3. Update import paths to use `doc_chunking.src`
4. Remove LLM and fuzzy matching code

## Examples

See `example.py` for complete usage examples.
