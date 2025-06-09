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
from .pdf_processor import extract_pdf_pages_into_images, get_pdf_document_object, close_pdf_document
from .docx_processor import extract_docx_content, extract_docx_structure, extract_docx_structure_with_naive_llm
from .document_analyzer import (
    extract_pdf_document_structure, 
    analyze_document_structure, 
    get_structure_summary
)

# Import layout detection module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.layout_detection import DocLayoutDetector



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
            pages = extract_pdf_pages_into_images(mock_file)
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


@app.post("/api/analyze-pdf-structure")
async def analyze_pdf_structure(file: UploadFile = File(...)):
    """Analyze PDF structure only (no image conversion)"""
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        file_content = await file.read()
        
        # Extract document structure only
        structure = extract_pdf_document_structure(file_content)
        
        # Get total pages from PDF
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_content, filetype="pdf")
        total_pages = len(doc)
        doc.close()
        
        return {
            "filename": file.filename,
            "file_type": file.content_type,
            "size": len(file_content),
            "structure": structure,
            "structure_summary": get_structure_summary(structure),
            "total_pages": total_pages,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing PDF structure: {str(e)}")


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


@app.post("/api/analyze-docx-with-naive-llm")
async def analyze_docx_with_naive_llm(file: UploadFile = File(...)):
    """Analyze DOCX structure using naive_llm method"""
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
        
        # Process with naive_llm
        result = extract_docx_structure_with_naive_llm(mock_file)
        
        if result["success"]:
            return {
                "filename": file.filename,
                "file_type": file.content_type,
                "success": True,
                "section_tree": result["section_tree"],
                "raw_text": result["raw_text"],
                "llm_annotated_text": result["llm_annotated_text"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DOCX with naive_llm: {str(e)}")


# Global layout detector instance (lazy loading)
_layout_detector = None

def get_layout_detector():
    """Get or create the layout detector instance"""
    global _layout_detector
    if _layout_detector is None:
        _layout_detector = DocLayoutDetector(model_name="docstructbench")
    return _layout_detector


@app.post("/api/detect-layout")
async def detect_layout(
    file: UploadFile = File(...),
    confidence: float = 0.25,
    model_name: str = "docstructbench",
    image_size: int = 1024
):
    """
    Detect document layout elements in an uploaded image or document.
    
    Args:
        file: Image file (PNG, JPG, etc.) or PDF file
        confidence: Confidence threshold for detections (0.0-1.0)
        model_name: Model to use ('docstructbench', 'd4la', 'doclaynet')
        image_size: Input image size for the model
    
    Returns:
        JSON with detected layout elements
    """
    try:
        # Validate file type
        allowed_image_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp"]
        allowed_document_types = ["application/pdf"]
        all_allowed_types = allowed_image_types + allowed_document_types
        
        if file.content_type not in all_allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. "
                      f"Supported types: {', '.join(all_allowed_types)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Initialize detector with requested model
        try:
            detector = DocLayoutDetector(
                model_name=model_name,
                confidence_threshold=confidence,
                image_size=image_size
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        results = []
        
        # Handle different file types
        if file.content_type in allowed_image_types:
            # Direct image processing
            # Save content to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Detect layout
                result = detector.detect(tmp_path)
                elements = result.get_elements()
                
                results.append({
                    "page_number": 0,
                    "elements": elements,
                    "total_elements": len(elements)
                })
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        
        elif file.content_type == "application/pdf":
            # Extract PDF pages as images and process each
            class MockUploadedFile:
                def __init__(self, content, content_type, name):
                    self._content = content
                    self.type = content_type
                    self.name = name
                
                def getvalue(self):
                    return self._content
            
            mock_file = MockUploadedFile(file_content, file.content_type, file.filename)
            pages = extract_pdf_pages(mock_file)
            
            if not pages:
                raise HTTPException(status_code=500, detail="Failed to extract PDF pages")
            
            # Process each page
            for page_num, page_img in enumerate(pages):
                # Save page to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    page_img.save(tmp_file.name, format='PNG')
                    tmp_path = tmp_file.name
                
                try:
                    # Detect layout on this page
                    result = detector.detect(tmp_path)
                    elements = result.get_elements()
                    
                    results.append({
                        "page_number": page_num,
                        "elements": elements,
                        "total_elements": len(elements)
                    })
                    
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_path)
        
        # Compile summary statistics
        total_elements = sum(page["total_elements"] for page in results)
        element_type_counts = {}
        
        for page in results:
            for element in page["elements"]:
                element_type = element["type"]
                element_type_counts[element_type] = element_type_counts.get(element_type, 0) + 1
        
        return {
            "filename": file.filename,
            "file_type": file.content_type,
            "model_used": model_name,
            "confidence_threshold": confidence,
            "total_pages": len(results),
            "total_elements": total_elements,
            "element_type_summary": element_type_counts,
            "pages": results,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting layout: {str(e)}")


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
        detector = DocLayoutDetector(
            model_name=model_name,
            confidence_threshold=confidence,
            image_size=image_size
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
            pages = extract_pdf_pages(mock_file)
            
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 