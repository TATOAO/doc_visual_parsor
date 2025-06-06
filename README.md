# Document Visual Parser

A powerful document analysis tool that extracts structure and content from PDF and DOCX files using AI-powered processing.

## Architecture

The application now uses a **microservices architecture** with separate backend and frontend components:

- **Backend**: FastAPI server that handles document processing, structure analysis, and content extraction
- **Frontend**: Streamlit web application that provides the user interface

## Features

- 📄 **PDF Processing**: Extract pages as images and analyze document structure
- 📝 **DOCX Processing**: Extract content and analyze document structure  
- 🔍 **Structure Analysis**: AI-powered detection of headings, sections, and document hierarchy
- 🖼️ **Visual Display**: Interactive PDF viewer with page navigation
- 🌐 **API-First**: RESTful API for integration with other applications
- 📊 **Document Info**: Detailed metadata and statistics

## Quick Start

### Option 1: Run Both Servers Together (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run both backend and frontend
python run_full_app.py
```

### Option 2: Run Servers Separately

**Terminal 1 - Backend API:**
```bash
python run_backend.py
```

**Terminal 2 - Frontend:**
```bash
python run_app.py
```

## Access Points

- **Frontend Application**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## API Endpoints

- `POST /api/upload-document` - Upload and process a document
- `POST /api/analyze-structure` - Analyze document structure only
- `POST /api/extract-pdf-pages` - Extract PDF pages as images
- `POST /api/extract-docx-content` - Extract DOCX content
- `GET /` - Health check

## Project Structure

```
doc_visual_parser/
├── backend/                 # FastAPI backend
│   ├── api_server.py       # Main FastAPI application
│   ├── document_analyzer.py # Document structure analysis
│   ├── pdf_processor.py    # PDF processing logic
│   └── docx_processor.py   # DOCX processing logic
├── frontend/               # Streamlit frontend
│   ├── app.py             # Main Streamlit application
│   ├── api_client.py      # Backend API client
│   ├── ui_components.py   # UI components
│   └── session_manager.py # Session state management
├── run_backend.py         # Backend server startup
├── run_app.py            # Frontend server startup
├── run_full_app.py       # Both servers startup
└── requirements.txt      # Dependencies
```

## Development

### Backend Development
```bash
# Run backend with auto-reload
python run_backend.py

# Or directly with uvicorn
uvicorn backend.api_server:app --reload --port 8000
```

### Frontend Development
```bash
# Run frontend (requires backend to be running)
python run_app.py

# Or directly with streamlit
streamlit run frontend/app.py --server.port 8501
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Streamlit**: Web app framework for data applications
- **PyMuPDF**: PDF processing and text extraction
- **python-docx**: DOCX document processing
- **Pillow**: Image processing
- **Requests**: HTTP client for API communication

## License

This project is licensed under the MIT License.


