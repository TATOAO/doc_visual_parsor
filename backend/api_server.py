from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import io
import base64
from PIL import Image
import tempfile
import os

# Import existing backend modules
from .pdf_processor import extract_pdf_pages, get_pdf_document_object, close_pdf_document
from .docx_processor import extract_docx_content, extract_docx_structure
from .document_analyzer import (
    extract_pdf_document_structure, 
    analyze_document_structure, 
    get_structure_summary
)

app = FastAPI(title="Document Visual Parser API", version="1.0.0")

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


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: PDF, DOCX"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Create a mock uploaded file object for existing functions
        class MockUploadedFile:
            def __init__(self, content, content_type, name):
                self._content = content
                self.type = content_type
                self.name = name
            
            def getvalue(self):
                return self._content
        
        mock_file = MockUploadedFile(file_content, file.content_type, file.filename)
        
        # Process based on file type
        result = {
            "filename": file.filename,
            "file_type": file.content_type,
            "size": len(file_content),
            "success": True
        }
        
        if file.content_type == "application/pdf":
            # Extract document structure
            structure = extract_pdf_document_structure(file_content)
            result["structure"] = structure
            result["structure_summary"] = get_structure_summary(structure)
            
            # Extract pages as images
            pages = extract_pdf_pages(mock_file)
            if pages:
                # Convert images to base64 for JSON response
                page_images = []
                for i, page_img in enumerate(pages):
                    # Convert PIL image to base64
                    img_buffer = io.BytesIO()
                    page_img.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    page_images.append({
                        "page_number": i,
                        "image_data": img_base64
                    })
                
                result["pages"] = page_images
                result["total_pages"] = len(pages)
            
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Extract DOCX content
            content = extract_docx_content(mock_file)
            structure = extract_docx_structure(mock_file)
            
            result["content"] = content
            result["structure"] = structure
            result["structure_summary"] = get_structure_summary(structure)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/api/analyze-structure")
async def analyze_structure(file: UploadFile = File(...)):
    """Analyze document structure only"""
    try:
        file_content = await file.read()
        
        class MockUploadedFile:
            def __init__(self, content, content_type, name):
                self.content = content
                self.type = content_type
                self.name = name
            
            def getvalue(self):
                return self.content
        
        mock_file = MockUploadedFile(file_content, file.content_type, file.filename)
        structure = analyze_document_structure(mock_file)
        
        return {
            "filename": file.filename,
            "structure": structure,
            "structure_summary": get_structure_summary(structure),
            "total_headings": len(structure)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing structure: {str(e)}")


@app.post("/api/extract-pdf-pages")
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
        pages = extract_pdf_pages(mock_file)
        
        if pages:
            page_images = []
            for i, page_img in enumerate(pages):
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
                "total_pages": len(pages)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 