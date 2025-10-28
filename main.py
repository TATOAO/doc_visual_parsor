#!/usr/bin/env python3
"""
Main entry point for the PDF Layout Extraction API service.

This script starts the FastAPI application server for the doc_visual_parsor project.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to start the FastAPI application."""
    try:
        logger.info("Starting PDF Layout Extraction API service...")
        
        # Import and run the FastAPI app
        import uvicorn
        from doc_chunking.app.app import app
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8887,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you're running from the project root and all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start the service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
