#!/usr/bin/env python3
"""
Document Visual Parser - Startup Script
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Main function to run the Streamlit app"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "frontend", "app.py")
        
        print("🚀 Starting Document Visual Parser...")
        print(f"📁 App location: {app_path}")
        print("🌐 Opening browser at: http://localhost:8501")
        print("---")
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 