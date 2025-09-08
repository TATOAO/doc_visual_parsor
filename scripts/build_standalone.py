#!/usr/bin/env python3
"""
Build script for creating a standalone executable of the document layout detection tool.

This script creates a lightweight standalone executable that includes only the necessary
dependencies, avoiding the large PyTorch overhead.
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
import argparse

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def create_requirements_file():
    """Create a minimal requirements file for the standalone build."""
    requirements = [
        "onnxruntime>=1.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "PyMuPDF>=1.20.0",  # For PDF support
        "pydantic>=2.0.0",  # For data validation
    ]
    
    requirements_file = Path("requirements_standalone.txt")
    with open(requirements_file, 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"Created requirements file: {requirements_file}")
    return requirements_file

def create_spec_file(script_path, output_name="doc_layout_detector"):
    """Create a PyInstaller spec file for the standalone build."""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{script_path}'],
    pathex=[],
    binaries=[],
    datas=[
        # Include any model files or data files here
        # ('path/to/model.onnx', '.'),
    ],
    hiddenimports=[
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        'cv2',
        'PIL',
        'PIL.Image',
        'numpy',
        'fitz',
        'pydantic',
        'pydantic.fields',
        'pydantic.main',
        'pydantic.types',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'torch',
        'torchvision',
        'ultralytics',
        'doclayout_yolo',
        'huggingface_hub',
        'transformers',
        'tensorflow',
        'keras',
        'sklearn',
        'matplotlib',
        'seaborn',
        'jupyter',
        'notebook',
        'ipython',
        'pandas',
        'scipy',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{output_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    spec_file = Path(f"{output_name}.spec")
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print(f"Created spec file: {spec_file}")
    return spec_file

def build_with_pyinstaller(spec_file, clean=True):
    """Build the executable using PyInstaller."""
    if clean and Path("build").exists():
        shutil.rmtree("build")
    if clean and Path("dist").exists():
        shutil.rmtree("dist")
    
    # Install PyInstaller if not available
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Build the executable
    cmd = [sys.executable, "-m", "PyInstaller", "--clean", str(spec_file)]
    run_command(cmd)
    
    print("Build completed successfully!")

def create_portable_package(output_name="doc_layout_detector"):
    """Create a portable package with the executable and necessary files."""
    dist_dir = Path("dist")
    exe_path = dist_dir / output_name
    
    if not exe_path.exists():
        print(f"Executable not found: {exe_path}")
        return
    
    # Create portable package directory
    package_dir = Path(f"{output_name}_portable")
    if package_dir.exists():
        shutil.rmtree(package_dir)
    
    package_dir.mkdir()
    
    # Copy executable
    if sys.platform == "win32":
        exe_name = f"{output_name}.exe"
    else:
        exe_name = output_name
    
    shutil.copy2(exe_path / exe_name, package_dir / exe_name)
    
    # Create README
    readme_content = f"""# {output_name.title()} - Portable Document Layout Detection Tool

This is a lightweight, standalone tool for document layout detection using ONNX Runtime.

## Usage

### Basic Usage
```bash
./{exe_name} --model model.onnx --input document.pdf --output results.json
```

### Batch Processing
```bash
./{exe_name} --model model.onnx --batch input_dir/ output_dir/
```

### Get Model Information
```bash
./{exe_name} --model model.onnx --info
```

## Requirements

- ONNX model file (converted from PyTorch using convert_to_onnx.py)
- Input files: PDF, PNG, JPG, JPEG, BMP, TIFF, TIF

## Model Conversion

To convert your PyTorch YOLO model to ONNX format:

```bash
python convert_to_onnx.py --model-path your_model.pt --output-path model.onnx
```

## Supported File Formats

- **Input**: PDF, PNG, JPG, JPEG, BMP, TIFF, TIF
- **Output**: JSON format with detected layout elements

## Performance

This tool uses ONNX Runtime instead of PyTorch, resulting in:
- ~90% smaller memory footprint
- Faster startup times
- No PyTorch dependencies
- Cross-platform compatibility

## Troubleshooting

If you encounter issues:
1. Ensure your ONNX model file is valid
2. Check that input files are in supported formats
3. Verify file permissions for input/output directories
"""
    
    with open(package_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Create example script
    example_script = f"""#!/bin/bash
# Example usage script for {output_name}

# Convert PyTorch model to ONNX (run this first)
# python convert_to_onnx.py --model-path your_model.pt --output-path model.onnx

# Detect layout for a single PDF
./{exe_name} --model model.onnx --input document.pdf --output results.json

# Process all PDFs in a directory
./{exe_name} --model model.onnx --batch input_dir/ output_dir/

# Get model information
./{exe_name} --model model.onnx --info
"""
    
    with open(package_dir / "example.sh", 'w') as f:
        f.write(example_script)
    
    # Make example script executable on Unix systems
    if sys.platform != "win32":
        os.chmod(package_dir / "example.sh", 0o755)
    
    print(f"Portable package created: {package_dir}")
    print(f"Executable size: {get_file_size(package_dir / exe_name):.1f} MB")

def get_file_size(file_path):
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)

def main():
    parser = argparse.ArgumentParser(description="Build standalone executable for document layout detection")
    parser.add_argument(
        "--script", 
        default="scripts/doc_layout_cli.py",
        help="Path to the main script to build (default: scripts/doc_layout_cli.py)"
    )
    parser.add_argument(
        "--output-name", 
        default="doc_layout_detector",
        help="Name of the output executable (default: doc_layout_detector)"
    )
    parser.add_argument(
        "--no-clean", 
        action="store_true",
        help="Don't clean build directories before building"
    )
    parser.add_argument(
        "--no-package", 
        action="store_true",
        help="Don't create portable package"
    )
    parser.add_argument(
        "--requirements-only", 
        action="store_true",
        help="Only create requirements file and exit"
    )
    
    args = parser.parse_args()
    
    # Validate script path
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        sys.exit(1)
    
    try:
        # Create requirements file
        requirements_file = create_requirements_file()
        
        if args.requirements_only:
            print("Requirements file created. Exiting.")
            return
        
        # Create spec file
        spec_file = create_spec_file(str(script_path), args.output_name)
        
        # Build executable
        build_with_pyinstaller(spec_file, clean=not args.no_clean)
        
        # Create portable package
        if not args.no_package:
            create_portable_package(args.output_name)
        
        print("\nBuild completed successfully!")
        print(f"Executable location: dist/{args.output_name}")
        
        if not args.no_package:
            print(f"Portable package: {args.output_name}_portable/")
        
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
