#!/usr/bin/env python3
"""
Build script for the doc-chunking library.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    
    # Directories to clean
    clean_dirs = ['build', 'dist', 'doc_chunking.egg-info', '__pycache__']
    
    for dir_name in clean_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}")
    
    # Clean __pycache__ directories recursively
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                full_path = os.path.join(root, dir_name)
                shutil.rmtree(full_path)
                print(f"Removed {full_path}")

def check_dependencies():
    """Check if build dependencies are installed."""
    print("Checking build dependencies...")
    
    required_packages = ['build', 'twine']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            print(f"Install with: pip install {package}")
            return False
    
    return True

def build_package():
    """Build the package."""
    print("Building package...")
    
    # Build the package
    if not run_command("python -m build", "Building package"):
        return False
    
    print("Package built successfully!")
    return True

def check_package():
    """Check the built package."""
    print("Checking package...")
    
    # Check the package with twine
    if not run_command("twine check dist/*", "Checking package"):
        return False
    
    print("Package check passed!")
    return True

def install_local():
    """Install the package locally for testing."""
    print("Installing package locally...")
    
    # Install in development mode
    if not run_command("pip install -e .", "Installing package locally"):
        return False
    
    print("Package installed locally!")
    return True

def test_import():
    """Test if the package can be imported."""
    print("Testing package import...")
    
    try:
        import doc_chunking
        print(f"✓ Successfully imported doc_chunking version {doc_chunking.__version__}")
        
        # Test basic functionality
        print("Testing basic functionality...")
        print(f"Available functions: {len(doc_chunking.__all__)} items")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import package: {e}")
        return False

def main():
    """Main build process."""
    print("=== Doc Chunking Library Build Script ===")
    
    # Check if we're in the right directory
    if not os.path.exists('pyproject.toml'):
        print("Error: pyproject.toml not found. Are you in the right directory?")
        sys.exit(1)
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if 'clean' in args:
        clean_build()
        return
    
    if 'check-deps' in args:
        if check_dependencies():
            print("All dependencies are installed!")
        else:
            print("Some dependencies are missing!")
        return
    
    # Default build process
    steps = [
        ("Clean build artifacts", clean_build),
        ("Check dependencies", check_dependencies),
        ("Build package", build_package),
        ("Check package", check_package),
    ]
    
    if 'install' in args:
        steps.append(("Install locally", install_local))
        steps.append(("Test import", test_import))
    
    # Execute steps
    for step_name, step_func in steps:
        print(f"\n--- {step_name} ---")
        if not step_func():
            print(f"❌ Failed at step: {step_name}")
            sys.exit(1)
        print(f"✅ Completed: {step_name}")
    
    print("\n=== Build Complete! ===")
    
    if os.path.exists('dist'):
        print("Built packages:")
        for file in os.listdir('dist'):
            print(f"  - dist/{file}")
    
    print("\nNext steps:")
    print("1. Test the package: python build.py install")
    print("2. Upload to PyPI: twine upload dist/*")

if __name__ == '__main__':
    main() 