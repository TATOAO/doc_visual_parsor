# Document Visual Parser - Frontend

A Streamlit-based web application for document analysis and structure parsing.

## Features

- 📄 **Document Upload**: Support for PDF and DOCX files
- 🖼️ **Visual Display**: High-quality document rendering with page navigation
- 📑 **Structure Analysis**: Left sidebar for chapter/section navigation (ready for AI integration)
- ⚙️ **Control Panel**: Tools for document processing and visualization options
- 🎨 **Modern UI**: Clean, responsive interface with custom styling

## Screenshots

The application features:
- **Left Sidebar**: Document structure and chapter navigation
- **Center Panel**: Document viewer with page navigation
- **Right Panel**: Controls and processing options

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Option A - Using the startup script:
```bash
python run_app.py
```

Option B - Direct Streamlit command:
```bash
streamlit run frontend/app.py
```

### 3. Open Your Browser

Navigate to: `http://localhost:8501`

## Usage

1. **Upload Document**: Click "Choose a document" or drag & drop a PDF/DOCX file
2. **View Content**: Document will be displayed in the center panel
3. **Navigate Pages**: For PDFs, use Previous/Next buttons to navigate
4. **Future Features**: Chapter structure will appear in the left sidebar once AI parsing is integrated

## File Structure

```
frontend/
├── app.py              # Main Streamlit application
└── README.md           # This file

requirements.txt        # Python dependencies
run_app.py             # Startup script
```

## Dependencies

- **Streamlit**: Web framework for the interface
- **PyMuPDF**: PDF processing and rendering
- **python-docx**: DOCX file processing
- **Pillow**: Image processing
- **pathlib2**: File path utilities

## Next Steps

This frontend is ready for integration with:
- DocLayout-YOLO for layout detection
- Multi-modal AI models for content analysis
- Chapter/section extraction algorithms
- Export and visualization features

## Development

To extend the application:

1. **Add AI Models**: Integrate layout detection in the document processing functions
2. **Enhance Sidebar**: Populate the chapters list with detected document structure
3. **Add Visualizations**: Implement bounding box overlays and confidence scores
4. **Export Features**: Add document extraction and export functionality

## Supported Formats

- **PDF**: Full page rendering with navigation
- **DOCX**: Text content extraction and display
- **Future**: Images, multi-page TIFF, and other document formats 