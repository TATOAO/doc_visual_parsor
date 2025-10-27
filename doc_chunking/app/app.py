"""
FastAPI application for PDF layout extraction using PdfStyleCVMixLayoutExtractor.

This app provides a REST API endpoint to upload PDF files and extract their layout
information using the hybrid CV + PDF approach.
"""

import os
import tempfile
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from doc_chunking.merging import PdfStyleCVMixLayoutExtractor
from doc_chunking.schemas import LayoutExtractionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Layout Extraction API",
    description="Extract layout information from PDF documents using hybrid CV + PDF approach",
    version="1.0.0"
)

# Global extractor instance (initialized once)
extractor: Optional[PdfStyleCVMixLayoutExtractor] = None


class ExtractionResponse(BaseModel):
    """Response model for layout extraction."""
    success: bool = Field(description="Whether the extraction was successful")
    message: str = Field(description="Status message")
    data: Optional[dict] = Field(description="Extracted layout data", default=None)
    metadata: Optional[dict] = Field(description="Extraction metadata", default=None)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether the model is loaded")


def initialize_extractor():
    """Initialize the PDF layout extractor."""
    global extractor
    try:
        logger.info("Initializing PdfStyleCVMixLayoutExtractor...")
        
        # Use the default model path from the project
        model_path = os.path.join(
            os.path.dirname(__file__),
            'model_parameters', 'layout_detection',
            'docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx'
        )
        
        # Fallback to relative path if absolute doesn't exist
        if not os.path.exists(model_path):
            model_path = 'model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx'
        
        extractor = PdfStyleCVMixLayoutExtractor(
            model_path=model_path,
            cv_confidence_threshold=0.1,  # Lower threshold for better detection
            cv_image_size=1024,
            cv_pdf_dpi=150,
            device="auto"
        )
        
        logger.info("PdfStyleCVMixLayoutExtractor initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the extractor when the app starts."""
    success = initialize_extractor()
    if not success:
        logger.warning("Failed to initialize extractor on startup")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if extractor is not None else "unhealthy",
        model_loaded=extractor is not None
    )


@app.post("/extract-layout", response_model=ExtractionResponse)
async def extract_layout(
    file: UploadFile = File(..., description="PDF file to process"),
    confidence_threshold: Optional[float] = Form(None, description="Confidence threshold for detection (0.0-1.0)"),
    max_pages: Optional[int] = Form(None, description="Maximum number of pages to process")
):
    """
    Extract layout information from uploaded PDF file.
    
    Args:
        file: PDF file to process
        confidence_threshold: Optional confidence threshold override
        max_pages: Optional maximum pages to process
        
    Returns:
        JSON response with extracted layout data
    """
    if extractor is None:
        raise HTTPException(
            status_code=503, 
            detail="Layout extractor not initialized. Please check server logs."
        )
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Create temporary file to store uploaded PDF
    temp_file = None
    try:
        # Read file content
        content = await file.read()
        
        # Validate PDF content
        if not content.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file format"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            temp_file = tmp_file.name
        
        logger.info(f"Processing PDF file: {file.filename} ({len(content)} bytes)")
        
        # Extract layout
        result: LayoutExtractionResult = extractor.detect_layout(
            input_data=temp_file,
            confidence_threshold=confidence_threshold,
            max_pages=max_pages
        )
        
        # Convert result to dictionary
        result_dict = result.model_dump()
        
        logger.info(f"Successfully processed {file.filename}: {len(result.elements)} elements extracted")
        
        return ExtractionResponse(
            success=True,
            message=f"Successfully extracted layout from {file.filename}",
            data=result_dict,
            metadata={
                "filename": file.filename,
                "file_size_bytes": len(content),
                "elements_count": len(result.elements),
                "pages_processed": result.metadata.get('pages_processed', 'unknown') if result.metadata else 'unknown',
                "extraction_method": result.metadata.get('extraction_method', 'unknown') if result.metadata else 'unknown'
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Layout extraction failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Layout extraction failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PDF Layout Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "extract_layout": "/extract-layout",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# python -m doc_chunking.app.app
if __name__ == "__main__":
    import uvicorn
    
    # Initialize extractor before starting server
    if initialize_extractor():
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            "doc_chunking.app.app:app",
            host="0.0.0.0",
            port=8887,
            reload=False,
            log_level="info"
        )
    else:
        logger.error("Failed to initialize extractor. Exiting.")
        exit(1)
