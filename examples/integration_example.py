"""
Example: Integrating doc_chunking FastAPI router into your own application.

This example demonstrates how to mount the doc_chunking API routes
into your own FastAPI application as a modular component.
"""

from fastapi import FastAPI
from doc_chunking import fastapi_app

# Create your main application
app = FastAPI(
    title="My Custom Application",
    description="An example application that includes document processing capabilities",
    version="1.0.0"
)

# Mount the document processing routes under a specific path
app.mount("/api/documents", fastapi_app)

# Add your own routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to My Custom Application",
        "doc_processing_available": True,
        "doc_processing_endpoints": "Available at /api/documents/"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "services": ["main", "doc_processing"]}

@app.get("/info")
async def app_info():
    return {
        "app_name": "My Custom Application",
        "features": [
            "Document processing (PDF, DOCX)",
            "AI-powered layout detection",
            "Intelligent chunking",
            "Custom business logic"
        ],
        "doc_processing_endpoints": [
            "/api/documents/api/chunk-document",
            "/api/documents/api/extract-pdf-pages-into-images",
            "/api/documents/api/extract-docx-content",
            "/api/documents/api/visualize-layout",
            "/api/documents/api/chunk-document-sse"
        ]
    }

# Example of using the doc_chunking library directly in your code
@app.post("/api/process-document")
async def process_document_custom():
    """
    Example endpoint that uses doc_chunking library directly
    rather than mounting the router.
    """
    from doc_chunking import Chunker
    
    # You can use the components directly in your business logic
    chunker = Chunker()
    
    return {
        "message": "This endpoint shows how to use doc_chunking components directly",
        "chunker_available": True,
        "note": "In a real implementation, you would process uploaded files here"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Custom Application with Document Processing...")
    print("📍 Main app: http://localhost:8000")
    print("📄 Document processing: http://localhost:8000/api/documents")
    print("📖 API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 