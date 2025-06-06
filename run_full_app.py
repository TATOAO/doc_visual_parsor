#!/usr/bin/env python3
"""
Document Visual Parser - Full Application Startup Script
Run this script to start both backend API and frontend Streamlit servers
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
    """Run the frontend Streamlit server"""
    global frontend_process
    try:
        # Wait a bit for backend to start
        time.sleep(3)
        
        print("🚀 Starting frontend Streamlit server...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "frontend", "app.py")
        
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
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
    
    print("🚀 Starting Document Visual Parser - Full Application")
    print("=" * 60)
    print("📍 Backend API will be available at: http://localhost:8000")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("📖 API documentation will be available at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop both servers")
    print("=" * 60)
    
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