#!/usr/bin/env python3
"""
Script to run the Document Visual Parser backend API server
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run the FastAPI backend server"""
    print("🚀 Starting Document Visual Parser Backend API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "backend.api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 