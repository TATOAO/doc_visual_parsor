#!/usr/bin/env python3
"""
Script to run the Document Visual Parser backend API server
"""

import uvicorn
import sys
import os
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration based on environment"""
    # Get log level from environment variable, default to INFO for production
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if log_level not in valid_levels:
        log_level = 'INFO'
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup basic logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Optionally add file handler for production
            # logging.FileHandler('app.log') if log_level != 'DEBUG' else None
        ]
    )
    
    # Configure specific loggers
    # Reduce noise from uvicorn access logs in production
    if log_level != 'DEBUG':
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Set application logger
    logger = logging.getLogger("doc_chunking")
    logger.info(f"Logging configured at {log_level} level")
    
    return logger

def main():
    """Run the FastAPI backend server"""
    logger = setup_logging()
    
    print("🚀 Starting Document Visual Parser Backend API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Get environment settings
        env = os.getenv('ENVIRONMENT', 'development')
        reload_enabled = env == 'development'
        
        logger.info(f"Starting server in {env} mode")
        logger.info(f"Auto-reload: {reload_enabled}")
        
        uvicorn.run(
            "backend.api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=reload_enabled,  # Only enable reload in development
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\n👋 Backend server stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 