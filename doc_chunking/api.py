from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path
from sse_starlette.sse import EventSourceResponse
import json

# Import from the same package
from .layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor
from .documents_chunking.chunker import Chunker
from .schemas.schemas import Section
from .utils.helper import remove_circular_references

# Import processors from the same package
from .processors.pdf_processor import extract_pdf_pages_into_images
from .processors.docx_processor import extract_docx_content

app = FastAPI(title="Document Visual Parser API", version="1.0.0")
router = APIRouter()

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for session data (in production, use Redis or database)
sessions = {}


def detect_file_type(file: UploadFile) -> str:
    """
    Detect file type from content_type with fallback to file extension.
    
    Args:
        file: UploadFile object
        
    Returns:
        Detected MIME type
    """
    # If content type is already specific and valid, use it
    if file.content_type and file.content_type not in [
        "application/octet-stream", 
        "binary/octet-stream"
    ]:
        return file.content_type
    
    # Fallback to file extension detection
    if file.filename:
        filename_lower = file.filename.lower()
        if filename_lower.endswith('.pdf'):
            return "application/pdf"
        elif filename_lower.endswith('.docx'):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif filename_lower.endswith('.doc'):
            # Note: .doc files are not supported, but we can detect them
            return "application/msword"
    
    # Return original content type if no extension match
    return file.content_type or "application/octet-stream"


@router.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Document Visual Parser API", "status": "running"}


@router.post("/api/extract-pdf-pages-into-images")
async def extract_pdf_pages_endpoint(file: UploadFile = File(...)):
    """Extract PDF pages as images"""
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_content = await file.read()
        
        class MockUploadedFile:
            def __init__(self, content, content_type, name):
                self.content = content
                self.type = content_type
                self.name = name
            
            def getvalue(self):
                return self.content
        
        mock_file = MockUploadedFile(file_content, file.content_type, file.filename)
        page_images = extract_pdf_pages_into_images(mock_file)
        
        if page_images:
            page_images_result = []
            for i, page_img in enumerate(page_images):
                img_buffer = io.BytesIO()
                page_img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                page_images_result.append({
                    "page_number": i,
                    "image_data": img_base64
                })
            
            return {
                "filename": file.filename,
                "pages": page_images_result,
                "total_pages": len(page_images_result)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to extract PDF pages")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF pages: {str(e)}")


@router.post("/api/extract-docx-content")
async def extract_docx_content_endpoint(file: UploadFile = File(...)):
    """Extract DOCX content"""
    try:
        if file.content_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raise HTTPException(status_code=400, detail="Only DOCX files are supported")
        
        file_content = await file.read()
        
        class MockUploadedFile:
            def __init__(self, content, content_type, name):
                self.content = content
                self.type = content_type
                self.name = name
            
            def getvalue(self):
                return self.content
        
        mock_file = MockUploadedFile(file_content, file.content_type, file.filename)
        content = extract_docx_content(mock_file)
        
        return {
            "filename": file.filename,
            "content": content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting DOCX content: {str(e)}")


@router.post("/api/visualize-layout")
async def visualize_layout(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    model_name: str = "docstructbench",
    image_size: int = 1024,
    page_number: int = 0
):
    """
    Detect layout and return annotated images.
    
    Args:
        file: Image file or PDF file
        confidence: Confidence threshold for detections
        model_name: Model to use
        image_size: Input image size for the model
        page_number: Page number to process (for PDFs, -1 for all pages)
    
    Returns:
        JSON with annotated images as base64
    """
    try:
        # Similar validation as detect_layout
        allowed_image_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp"]
        allowed_document_types = ["application/pdf"]
        all_allowed_types = allowed_image_types + allowed_document_types
        
        if file.content_type not in all_allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        file_content = await file.read()
        
        # Initialize detector
        detector = PdfStyleCVMixLayoutExtractor(
            cv_model_name=model_name,
            cv_confidence_threshold=confidence,
            cv_image_size=image_size
        )
        
        annotated_images = []
        
        if file.content_type in allowed_image_types:
            # Process single image
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Detect and visualize
                result = detector.detect(tmp_path)
                annotated_img = detector.visualize(tmp_path, result)
                
                # Convert to base64
                img_pil = Image.fromarray(annotated_img)
                img_buffer = io.BytesIO()
                img_pil.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                annotated_images.append({
                    "page_number": 0,
                    "annotated_image": img_base64,
                    "total_elements": len(result.get_elements())
                })
                
            finally:
                os.unlink(tmp_path)
        
        elif file.content_type == "application/pdf":
            # Process PDF pages
            class MockUploadedFile:
                def __init__(self, content, content_type, name):
                    self._content = content
                    self.type = content_type
                    self.name = name
                
                def getvalue(self):
                    return self._content
            
            mock_file = MockUploadedFile(file_content, file.content_type, file.filename)
            pages = extract_pdf_pages_into_images(mock_file)
            
            if not pages:
                raise HTTPException(status_code=500, detail="Failed to extract PDF pages")
            
            # Process specified page(s)
            pages_to_process = range(len(pages)) if page_number == -1 else [page_number]
            
            for page_num in pages_to_process:
                if page_num >= len(pages):
                    continue
                    
                page_img = pages[page_num]
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    page_img.save(tmp_file.name, format='PNG')
                    tmp_path = tmp_file.name
                
                try:
                    # Detect and visualize
                    result = detector.detect(tmp_path)
                    annotated_img = detector.visualize(tmp_path, result)
                    
                    # Convert to base64
                    img_pil = Image.fromarray(annotated_img)
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    annotated_images.append({
                        "page_number": page_num,
                        "annotated_image": img_base64,
                        "total_elements": len(result.get_elements())
                    })
                    
                finally:
                    os.unlink(tmp_path)
        
        return {
            "filename": file.filename,
            "model_used": model_name,
            "total_pages": len(annotated_images),
            "annotated_pages": annotated_images,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error visualizing layout: {str(e)}")


@router.post("/api/chunk-document")
async def chunk_document(file: UploadFile = File(...)):
    """Chunk a PDF or DOCX document and return the section tree."""
    try:
        # Use improved file type detection
        detected_type = detect_file_type(file)
        
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if detected_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {detected_type}. Supported types: PDF, DOCX. Original content_type: {file.content_type}"
            )
        file_content = await file.read()
        # Save to a temporary file for compatibility with chunker
        suffix = ".pdf" if detected_type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        try:
            chunker = Chunker()
            # Use async version to avoid asyncio.run() in existing event loop
            sections = []
            async for section in chunker.chunk_async(tmp_path):
                sections.append(section)
            
            # Combine sections into single result like the sync version does
            if not sections:
                section_result = Section(title="Empty Document", content="", level=0)
            elif len(sections) == 1:
                section_result = sections[0]
            else:
                # If multiple sections, create a root section containing them
                section_result = Section(
                    title="Document",
                    content="",
                    level=0,
                    sub_sections=sections
                )
            
            remove_circular_references(section_result)
            return {
                "filename": file.filename,
                "file_type": detected_type,
                "original_content_type": file.content_type,
                "success": True,
                "section_tree": section_result.model_dump()
            }
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document: {str(e)}")


@router.post("/api/chunk-document-sse")
async def chunk_document_sse(file: UploadFile = File(...)):
    """
    Chunk a PDF or DOCX document and stream the section tree as SSE events.
    """
    try:
        # Use improved file type detection
        detected_type = detect_file_type(file)
        
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if detected_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {detected_type}. Supported types: PDF, DOCX. Original content_type: {file.content_type}"
            )
        file_content = await file.read()
        suffix = ".pdf" if detected_type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        async def event_generator():
            try:
                chunker = Chunker()
                async for section in chunker.chunk_async(tmp_path):
                    remove_circular_references(section)
                    yield {
                        "event": "section",
                        "data": json.dumps(section.model_dump(), ensure_ascii=False)
                    }
                yield {
                    "event": "end",
                    "data": json.dumps({"success": True})
                }
            finally:
                os.unlink(tmp_path)

        return EventSourceResponse(event_generator())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document (SSE): {str(e)}")



app.include_router(router)

def run_server():
    """Entry point for running the server via console script"""
    import argparse
    import uvicorn
    import logging
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run the doc-chunking FastAPI server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info', 
                        choices=['critical', 'error', 'warning', 'info', 'debug', 'trace'],
                        help='Log level (default: info)')
    
    args = parser.parse_args()
    
    # Setup logging
    from .utils.logging_config import setup_logging
    
    # Get environment settings (can be overridden by CLI args)
    log_level = os.getenv('LOG_LEVEL', args.log_level.upper()).upper()
    env = os.getenv('ENVIRONMENT', 'development')
    
    # Configure logging
    setup_logging(level=log_level, use_colors=True)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Document Visual Parser API Server...")
    logger.info(f"📍 Server will be available at: http://{args.host}:{args.port}")
    logger.info(f"📖 API documentation will be available at: http://{args.host}:{args.port}/docs")
    logger.info(f"🔄 Environment: {env}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        # Use reload option from CLI args or default to environment setting
        reload_enabled = args.reload or (env == 'development')
        
        # Use import string format when reload is enabled
        if reload_enabled:
            uvicorn.run(
                "doc_chunking.api:app",
                host=args.host,
                port=args.port,
                log_level=args.log_level.lower(),
                reload=True
            )
        else:
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level=args.log_level.lower(),
                reload=False
            )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        raise


if __name__ == "__main__":
    run_server()
