#!/usr/bin/env python3
"""
Command-line script to run the doc-chunking FastAPI server.
"""

import argparse
import uvicorn
import sys
import os

# Add the parent directory to the path so we can import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def main():
    """Main entry point for the doc-chunking server."""
    parser = argparse.ArgumentParser(description='Run the doc-chunking FastAPI server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info', choices=['critical', 'error', 'warning', 'info', 'debug', 'trace'],
                        help='Log level (default: info)')
    
    args = parser.parse_args()
    
    try:
        # Import the FastAPI app
        from backend.api_server import app
        
        print(f"Starting doc-chunking server on {args.host}:{args.port}")
        print(f"API documentation available at: http://{args.host}:{args.port}/docs")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level
        )
    except ImportError as e:
        print(f"Error: Could not import backend API server: {e}")
        print("Make sure you have installed the package with backend dependencies.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 