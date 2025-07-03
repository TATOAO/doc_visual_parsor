# Doc Chunking

A powerful document analysis and chunking library for PDF and DOCX files using AI-powered processing.

## Features

- **Document Processing**: Extract content from PDF and DOCX files
- **Layout Detection**: AI-powered layout analysis using computer vision
- **Intelligent Chunking**: Smart document segmentation based on structure
- **FastAPI Integration**: Built-in REST API for easy integration
- **Modular Design**: Easy to integrate into existing applications

## Installation

```bash
pip install doc-chunking
```

For development:
```bash
git clone <repository-url>
cd doc_chunking
pip install -e .
```

## Quick Start

### Using the FastAPI Server

Run the built-in server:
```bash
doc-chunking-server
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`.

### Using as a Library

```python
from doc_chunking import Chunker, fastapi_app

# Use the chunker directly
chunker = Chunker()
sections = chunker.chunk("path/to/document.pdf")

# Or integrate the FastAPI app into your own application
from fastapi import FastAPI
from doc_chunking import fastapi_app

app = FastAPI()
app.mount("/doc-processing", fastapi_app)
```

### Integration with Existing FastAPI Applications

```python
from fastapi import FastAPI
from doc_chunking import fastapi_app

# Your main application
app = FastAPI(title="My Application")

# Mount the document processing routes
app.mount("/api/docs", fastapi_app)

# Your existing routes
@app.get("/")
def root():
    return {"message": "My Application"}
```

## API Endpoints

- `GET /` - Health check
- `POST /api/chunk-document` - Chunk a document and return section tree
- `POST /api/chunk-document-sse` - Stream chunking results via Server-Sent Events
- `POST /api/extract-pdf-pages-into-images` - Extract PDF pages as images
- `POST /api/extract-docx-content` - Extract DOCX content
- `POST /api/visualize-layout` - Visualize document layout with AI detection

## Library Usage

### Basic Document Chunking

```python
from doc_chunking import Chunker

chunker = Chunker()

# Synchronous chunking
sections = chunker.chunk("document.pdf")

# Asynchronous chunking
import asyncio
async def process_document():
    async for section in chunker.chunk_async("document.pdf"):
        print(f"Section: {section.title}")
        print(f"Content: {section.content}")

asyncio.run(process_document())
```

### Document Processing

```python
from doc_chunking import extract_pdf_pages_into_images, extract_docx_content

# Extract PDF pages as images
with open("document.pdf", "rb") as f:
    images = extract_pdf_pages_into_images(f)

# Extract DOCX content
with open("document.docx", "rb") as f:
    content = extract_docx_content(f)
```

### Layout Detection

```python
from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor

detector = PdfStyleCVMixLayoutExtractor()
layout_result = detector.detect("document.pdf")
```

## Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `ENVIRONMENT`: Set to 'production' or 'development'

### Logging Configuration

```python
from doc_chunking.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(level="INFO", log_file="app.log")
```

## Dependencies

- FastAPI for REST API
- PyMuPDF for PDF processing
- python-docx for DOCX processing
- OpenAI for LLM-based processing
- Computer vision models for layout detection
- Pydantic for data validation

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd doc_chunking

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Running the Server in Development

```bash
# Using the console script
doc-chunking-server

# Or directly with uvicorn
uvicorn doc_chunking.api:app --reload
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests.

## Support

For issues and questions, please use the GitHub issue tracker.
