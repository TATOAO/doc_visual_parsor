from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Fix: Use absolute imports for backend modules
from backend.pdf_processor import extract_pdf_pages_into_images
from backend.docx_processor import extract_docx_content

# Import layout detection module
sys.path.append(str(Path(__file__).parent.parent))
from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor

# --- Chunker import ---
from doc_chunking.documents_chunking.chunker import Chunker
from doc_chunking.naive_llm.helpers.section_token_parsor import remove_circular_references

app = FastAPI(title="Document Visual Parser API", version="1.0.0")


"""
app.post("/api/upload-document") # upload and process a document
app.post("/api/analyze-structure") # analyze document structure only
app.post("/api/analyze-pdf-structure") # analyze PDF structure only (no image conversion)
app.post("/api/extract-pdf-pages-into-images") # extract PDF pages as images
app.post("/api/extract-docx-content") # extract DOCX content

"""


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


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Document Visual Parser API", "status": "running"}


@app.post("/api/extract-pdf-pages-into-images")
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
            page_images = []
            for i, page_img in enumerate(page_images):
                img_buffer = io.BytesIO()
                page_img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                page_images.append({
                    "page_number": i,
                    "image_data": img_base64
                })
            
            return {
                "filename": file.filename,
                "pages": page_images,
                "total_pages": len(page_images)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to extract PDF pages")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF pages: {str(e)}")


@app.post("/api/extract-docx-content")
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


@app.post("/api/visualize-layout")
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


@app.post("/api/chunk-document")
async def chunk_document(file: UploadFile = File(...)):
    """Chunk a PDF or DOCX document and return the section tree."""
    try:
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, DOCX"
            )
        file_content = await file.read()
        # Save to a temporary file for compatibility with chunker
        suffix = ".pdf" if file.content_type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        try:
            chunker = Chunker()
            section_result = chunker.chunk(tmp_path)
            remove_circular_references(section_result)
            return {
                "filename": file.filename,
                "file_type": file.content_type,
                "success": True,
                "section_tree": section_result.model_dump()
            }
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document: {str(e)}")


# curl -X POST http://localhost:8000/api/chunk-document-sse -F "file=@tests/test_data/1-1 买卖合同（通用版）.docx;type=application/vnd.openxmlformats-officedocument.wordprocessingml.document"
@app.post("/api/chunk-document-sse")
async def chunk_document_sse(file: UploadFile = File(...)):
    """
    Chunk a PDF or DOCX document and stream the section tree as SSE events.
    """
    try:
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, DOCX"
            )
        file_content = await file.read()
        suffix = ".pdf" if file.content_type == "application/pdf" else ".docx"
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

# python -m backend.api_server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 