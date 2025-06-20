#!/usr/bin/env python3
"""
Document Chunking Frontend - Full Application Startup Script
Run this script to start both backend API and frontend Next.js servers
"""

import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path

# Global variables to track processes
backend_process = None
frontend_process = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down servers...")
    
    if backend_process:
        print("🔴 Stopping backend server...")
        backend_process.terminate()
        backend_process.wait()
    
    if frontend_process:
        print("🔴 Stopping frontend server...")
        frontend_process.terminate()
        frontend_process.wait()
    
    print("👋 All servers stopped")
    sys.exit(0)

def run_backend():
    """Run the backend API server"""
    global backend_process
    try:
        print("🚀 Starting backend API server...")
        backend_process = subprocess.Popen([
            sys.executable, "run_backend.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor backend output
        for line in iter(backend_process.stdout.readline, ''):
            if line:
                print(f"[BACKEND] {line.strip()}")
        
    except Exception as e:
        print(f"❌ Error running backend: {e}")

def run_frontend():
    """Run the frontend Next.js server"""
    global frontend_process
    try:
        # Wait a bit for backend to start
        time.sleep(3)
        
        print("🚀 Starting frontend Next.js server...")
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "doc-chunking-ui")
        
        # Check if node_modules exists, if not run npm install
        if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        
        frontend_process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Monitor frontend output
        for line in iter(frontend_process.stdout.readline, ''):
            if line:
                print(f"[FRONTEND] {line.strip()}")
        
    except Exception as e:
        print(f"❌ Error running frontend: {e}")

def main():
    """Main function to run both servers"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Document Chunking Visualizer - Full Application")
    print("=" * 65)
    print("📍 Backend API will be available at: http://localhost:8000")
    print("🌐 Frontend will be available at: http://localhost:3000")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop both servers")
    print("=" * 65)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Start frontend in a separate thread
        frontend_thread = threading.Thread(target=run_frontend, daemon=True)
        frontend_thread.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process and backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
                
            if frontend_process and frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ Error running application: {e}")
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 