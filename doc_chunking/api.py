from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import io
import base64
from PIL import Image
import tempfile
import os
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

# Pydantic models for API documentation
class HealthCheckResponse(BaseModel):
    """Health check response model"""
    message: str = Field(..., description="API status message")
    status: str = Field(..., description="Current status of the API")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Document Visual Parser API",
                "status": "running"
            }
        }

class PageImageData(BaseModel):
    """Model for PDF page image data"""
    page_number: int = Field(..., description="Page number (0-indexed)")
    image_data: str = Field(..., description="Base64 encoded image data")

class ExtractPdfPagesResponse(BaseModel):
    """Response model for PDF page extraction"""
    filename: str = Field(..., description="Original filename")
    pages: List[PageImageData] = Field(..., description="List of extracted page images")
    total_pages: int = Field(..., description="Total number of pages extracted")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "document.pdf",
                "pages": [
                    {
                        "page_number": 0,
                        "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    }
                ],
                "total_pages": 1
            }
        }

class ExtractDocxContentResponse(BaseModel):
    """Response model for DOCX content extraction"""
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Extracted text content from DOCX")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "document.docx",
                "content": "This is the extracted text content from the DOCX file..."
            }
        }

class AnnotatedPageData(BaseModel):
    """Model for annotated page data"""
    page_number: int = Field(..., description="Page number (0-indexed)")
    annotated_image: str = Field(..., description="Base64 encoded annotated image")
    total_elements: int = Field(..., description="Number of layout elements detected")

class VisualizeLayoutResponse(BaseModel):
    """Response model for layout visualization"""
    filename: str = Field(..., description="Original filename")
    model_used: str = Field(..., description="Model used for layout detection")
    total_pages: int = Field(..., description="Total number of pages processed")
    annotated_pages: List[AnnotatedPageData] = Field(..., description="List of annotated pages")
    success: bool = Field(..., description="Whether the operation was successful")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "document.pdf",
                "model_used": "docstructbench",
                "total_pages": 1,
                "annotated_pages": [
                    {
                        "page_number": 0,
                        "annotated_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        "total_elements": 5
                    }
                ],
                "success": True
            }
        }

class ChunkDocumentResponse(BaseModel):
    """Response model for document chunking"""
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Detected file type")
    original_content_type: Optional[str] = Field(None, description="Original content type from upload")
    success: bool = Field(..., description="Whether the operation was successful")
    section_tree: Dict[str, Any] = Field(..., description="Hierarchical section structure")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "document.pdf",
                "file_type": "application/pdf",
                "original_content_type": "application/pdf",
                "success": True,
                "section_tree": {
                    "title": "Document Title",
                    "content": "Document content...",
                    "level": 0,
                    "sub_sections": []
                }
            }
        }

class FlattenedSection(BaseModel):
    """Model for flattened section data"""
    title: str = Field(..., description="Title joined by all parent titles with format 'titleA-titleA1-...'")
    content: str = Field(..., description="Content of the section")
    level: int = Field(..., description="Level of the section")

class ChunkDocumentFlatResponse(BaseModel):
    """Response model for flattened document chunking"""
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Detected file type")
    original_content_type: Optional[str] = Field(None, description="Original content type from upload")
    success: bool = Field(..., description="Whether the operation was successful")
    flattened_sections: List[FlattenedSection] = Field(..., description="List of flattened sections")
    total_sections: int = Field(..., description="Total number of sections")
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "document.pdf",
                "file_type": "application/pdf",
                "original_content_type": "application/pdf",
                "success": True,
                "flattened_sections": [
                    {
                        "title": "Chapter 1",
                        "content": "Chapter content...",
                        "level": 0
                    },
                    {
                        "title": "Chapter 1-Section 1.1",
                        "content": "Section content...",
                        "level": 1
                    }
                ],
                "total_sections": 2
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Unsupported file type"
            }
        }

app = FastAPI(
    title="Document Visual Parser API",
    version="1.0.0",
    description="API for parsing, chunking, and analyzing document layouts with support for PDF and DOCX files",
    contact={
        "name": "Document Parser Team",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)
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


@router.get(
    "/",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check if the API is running and healthy",
    tags=["Health"]
)
async def root() -> HealthCheckResponse:
    """
    Health check endpoint to verify API status.
    
    Returns:
        HealthCheckResponse: API status information
    """
    return HealthCheckResponse(
        message="Document Visual Parser API",
        status="running"
    )


@router.post(
    "/api/extract-pdf-pages-into-images",
    response_model=ExtractPdfPagesResponse,
    summary="Extract PDF Pages as Images",
    description="Extract all pages from a PDF file and return them as base64-encoded images",
    responses={
        200: {"description": "PDF pages extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or file format"},
        500: {"model": ErrorResponse, "description": "Internal server error during extraction"}
    },
    tags=["PDF Processing"]
)
async def extract_pdf_pages_endpoint(
    file: UploadFile = File(..., description="PDF file to extract pages from", media_type="application/pdf")
) -> ExtractPdfPagesResponse:
    """
    Extract all pages from a PDF file and return them as base64-encoded PNG images.
    
    Args:
        file: PDF file upload
        
    Returns:
        ExtractPdfPagesResponse: List of extracted page images with metadata
        
    Raises:
        HTTPException: If file type is not PDF or extraction fails
    """
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
                page_images_result.append(PageImageData(
                    page_number=i,
                    image_data=img_base64
                ))
            
            return ExtractPdfPagesResponse(
                filename=file.filename,
                pages=page_images_result,
                total_pages=len(page_images_result)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to extract PDF pages")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting PDF pages: {str(e)}")


@router.post(
    "/api/extract-docx-content",
    response_model=ExtractDocxContentResponse,
    summary="Extract DOCX Content",
    description="Extract text content from a DOCX file",
    responses={
        200: {"description": "DOCX content extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or file format"},
        500: {"model": ErrorResponse, "description": "Internal server error during extraction"}
    },
    tags=["DOCX Processing"]
)
async def extract_docx_content_endpoint(
    file: UploadFile = File(..., description="DOCX file to extract content from", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
) -> ExtractDocxContentResponse:
    """
    Extract text content from a DOCX file.
    
    Args:
        file: DOCX file upload
        
    Returns:
        ExtractDocxContentResponse: Extracted text content with metadata
        
    Raises:
        HTTPException: If file type is not DOCX or extraction fails
    """
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
        
        return ExtractDocxContentResponse(
            filename=file.filename,
            content=content
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting DOCX content: {str(e)}")


@router.post(
    "/api/visualize-layout",
    response_model=VisualizeLayoutResponse,
    summary="Visualize Document Layout",
    description="Detect layout elements in documents and return annotated images showing the detected elements",
    responses={
        200: {"description": "Layout visualization completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error during layout detection"}
    },
    tags=["Layout Detection"]
)
async def visualize_layout(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, GIF, BMP) or PDF file to analyze"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold for layout element detection"),
    model_name: str = Query("docstructbench", description="Model to use for layout detection"),
    image_size: int = Query(1024, ge=512, le=2048, description="Input image size for the model"),
    page_number: int = Query(0, ge=-1, description="Page number to process (for PDFs, -1 for all pages)")
) -> VisualizeLayoutResponse:
    """
    Detect layout elements in documents and return annotated images.
    
    This endpoint processes image files (PNG, JPEG, GIF, BMP) or PDF files and detects
    layout elements like text blocks, titles, figures, etc. It returns annotated images
    showing the detected elements with bounding boxes.
    
    Args:
        file: Image file or PDF file to analyze
        confidence: Confidence threshold for layout element detection (0.0-1.0)
        model_name: Model to use for layout detection
        image_size: Input image size for the model (512-2048)
        page_number: Page number to process (for PDFs, -1 for all pages)
    
    Returns:
        VisualizeLayoutResponse: Annotated images with layout elements highlighted
        
    Raises:
        HTTPException: If file type is not supported or processing fails
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
                
                annotated_images.append(AnnotatedPageData(
                    page_number=0,
                    annotated_image=img_base64,
                    total_elements=len(result.get_elements())
                ))
                
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
                    
                    annotated_images.append(AnnotatedPageData(
                        page_number=page_num,
                        annotated_image=img_base64,
                        total_elements=len(result.get_elements())
                    ))
                    
                finally:
                    os.unlink(tmp_path)
        
        return VisualizeLayoutResponse(
            filename=file.filename,
            model_used=model_name,
            total_pages=len(annotated_images),
            annotated_pages=annotated_images,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error visualizing layout: {str(e)}")


@router.post(
    "/api/chunk-document",
    response_model=ChunkDocumentResponse,
    summary="Chunk Document",
    description="Parse and chunk a PDF or DOCX document into hierarchical sections",
    responses={
        200: {"description": "Document chunked successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or file format"},
        500: {"model": ErrorResponse, "description": "Internal server error during chunking"}
    },
    tags=["Document Chunking"]
)
async def chunk_document(
    file: UploadFile = File(..., description="PDF or DOCX file to chunk into sections")
) -> ChunkDocumentResponse:
    """
    Parse and chunk a PDF or DOCX document into hierarchical sections.
    
    This endpoint processes PDF or DOCX files and extracts their content into a hierarchical
    structure of sections and subsections. It uses layout detection and content analysis
    to identify document structure.
    
    Args:
        file: PDF or DOCX file to process
        
    Returns:
        ChunkDocumentResponse: Hierarchical section structure with metadata
        
    Raises:
        HTTPException: If file type is not supported or processing fails
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
            return ChunkDocumentResponse(
                filename=file.filename,
                file_type=detected_type,
                original_content_type=file.content_type,
                success=True,
                section_tree=section_result.model_dump()
            )
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document: {str(e)}")


@router.post(
    "/api/chunk-document-sse",
    summary="Chunk Document (Server-Sent Events)",
    description="Parse and chunk a PDF or DOCX document into hierarchical sections, streaming results as SSE events",
    responses={
        200: {
            "description": "Document chunking stream started successfully",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": "Server-Sent Events stream containing section data and completion status"
                    }
                }
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid file type or file format"},
        500: {"model": ErrorResponse, "description": "Internal server error during chunking"}
    },
    tags=["Document Chunking"],
    response_class=EventSourceResponse
)
async def chunk_document_sse(
    file: UploadFile = File(..., description="PDF or DOCX file to chunk into sections")
) -> EventSourceResponse:
    """
    Parse and chunk a PDF or DOCX document into hierarchical sections, streaming results as SSE events.
    
    This endpoint processes PDF or DOCX files and streams the extracted sections as
    Server-Sent Events (SSE). This is useful for real-time progress updates and
    handling large documents.
    
    The SSE stream will emit:
    - 'section' events containing individual section data
    - 'end' event when processing is complete
    
    Args:
        file: PDF or DOCX file to process
        
    Returns:
        EventSourceResponse: SSE stream of section data
        
    Raises:
        HTTPException: If file type is not supported or processing fails
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
                previous_count = 0  # Track how many sections we've already sent
                
                async for flattened_sections in chunker.chunk_flat_async(tmp_path):
                    # Only send new sections that we haven't sent before
                    new_sections = flattened_sections[previous_count:]
                    
                    for section in new_sections:
                        yield {
                            "event": "section",
                            "data": json.dumps(section, ensure_ascii=False)
                        }
                    
                    # Update the count of sent sections
                    previous_count = len(flattened_sections)
                
                yield {
                    "event": "end",
                    "data": json.dumps({"success": True})
                }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e), "success": False})
                }
            finally:
                os.unlink(tmp_path)

        return EventSourceResponse(event_generator())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document (SSE): {str(e)}")


@router.post(
    "/api/flatten-document",
    response_model=ChunkDocumentFlatResponse,
    summary="Flatten Document",
    description="Parse and flatten a PDF or DOCX document into a single-level list of sections",
    responses={
        200: {"description": "Document flattened successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type or file format"},
        500: {"model": ErrorResponse, "description": "Internal server error during flattening"}
    },
    tags=["Document Chunking"]
)
async def flatten_document(
    file: UploadFile = File(..., description="PDF or DOCX file to flatten into sections")
) -> ChunkDocumentFlatResponse:
    """
    Parse and flatten a PDF or DOCX document into a single-level list of sections.
    
    This endpoint processes PDF or DOCX files and extracts their content into a
    single-level list of sections. It uses layout detection and content analysis
    to identify document structure.
    
    Args:
        file: PDF or DOCX file to process
        
    Returns:
        ChunkDocumentFlatResponse: Flattened section structure with metadata
        
    Raises:
        HTTPException: If file type is not supported or processing fails
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
        # Save to a temporary file for compatibility with chunker
        suffix = ".pdf" if detected_type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            chunker = Chunker()
            # Use async version to avoid asyncio.run() in existing event loop
            flattened_sections_list = []
            async for flattened_sections in chunker.chunk_flat_async(tmp_path):
                # Keep only the latest/most complete version
                flattened_sections_list = flattened_sections
            
            # Convert to FlattenedSection objects
            flattened_section_objects = [
                FlattenedSection(
                    title=section['title'],
                    content=section['content'],
                    level=section['level']
                )
                for section in flattened_sections_list
            ]

            return ChunkDocumentFlatResponse(
                filename=file.filename,
                file_type=detected_type,
                original_content_type=file.content_type,
                success=True,
                flattened_sections=flattened_section_objects,
                total_sections=len(flattened_section_objects)
            )
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error flattening document: {str(e)}")


@router.post(
    "/api/flatten-document-sse",
    summary="Flatten Document (Server-Sent Events)",
    description="Parse and flatten a PDF or DOCX document into a single-level list of sections, streaming results as SSE events",
    responses={
        200: {
            "description": "Document flattening stream started successfully",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": "Server-Sent Events stream containing flattened section data and completion status"
                    }
                }
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid file type or file format"},
        500: {"model": ErrorResponse, "description": "Internal server error during flattening"}
    },
    tags=["Document Chunking"],
    response_class=EventSourceResponse
)
async def flatten_document_sse(
    file: UploadFile = File(..., description="PDF or DOCX file to flatten into sections")
) -> EventSourceResponse:
    """
    Parse and flatten a PDF or DOCX document into a single-level list of sections, streaming results as SSE events.
    
    This endpoint processes PDF or DOCX files and streams the extracted sections as
    Server-Sent Events (SSE). This is useful for real-time progress updates and
    handling large documents.
    
    The SSE stream will emit:
    - 'section' events containing individual flattened section data
    - 'end' event when processing is complete
    
    Args:
        file: PDF or DOCX file to process
        
    Returns:
        EventSourceResponse: SSE stream of flattened section data
        
    Raises:
        HTTPException: If file type is not supported or processing fails
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
                previous_count = 0  # Track how many sections we've already sent
                
                async for flattened_sections in chunker.chunk_flat_async(tmp_path):
                    # Only send new sections that we haven't sent before
                    new_sections = flattened_sections[previous_count:]
                    
                    for section in new_sections:
                        yield {
                            "event": "section",
                            "data": json.dumps(section, ensure_ascii=False)
                        }
                    
                    # Update the count of sent sections
                    previous_count = len(flattened_sections)
                
                yield {
                    "event": "end",
                    "data": json.dumps({"success": True})
                }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e), "success": False})
                }
            finally:
                os.unlink(tmp_path)

        return EventSourceResponse(event_generator())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error flattening document (SSE): {str(e)}")


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
