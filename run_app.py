#!/usr/bin/env python3
"""
Document Visual Parser - Frontend Startup Script
Run this script to start the Streamlit frontend application

Note: This now requires the backend API server to be running separately.
Run 'python run_backend.py' in another terminal first.
"""

import subprocess
import sys
import os
import requests
import time

def check_backend_api():
    """Check if the backend API is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function to run the Streamlit app"""
    try:
        # Check if backend API is running
        print("🔍 Checking backend API connection...")
        if not check_backend_api():
            print("⚠️  Backend API is not running!")
            print("📋 Please start the backend server first:")
            print("   1. Open another terminal")
            print("   2. Run: python run_backend.py")
            print("   3. Wait for the server to start")
            print("   4. Then run this script again")
            print("")
            print("🔄 Or run both servers with: python run_full_app.py")
            return
        
        print("✅ Backend API is running")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "frontend", "app.py")
        
        print("🚀 Starting Document Visual Parser Frontend...")
        print(f"📁 App location: {app_path}")
        print("🌐 Frontend will be available at: http://localhost:8501")
        print("🔗 Backend API is running at: http://localhost:8000")
        print("---")
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Frontend application stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 