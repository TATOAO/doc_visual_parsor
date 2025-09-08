#!/usr/bin/env python3
"""
Convert PyTorch YOLO model to ONNX format for lightweight deployment.

This script converts the DocLayout-YOLO model from PyTorch format to ONNX format,
which can be used with ONNX Runtime for much smaller deployment size.
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from ultralytics import YOLO
    import onnx
    from onnxsim import simplify
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install torch ultralytics onnx onnxsim")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_yolo_to_onnx(
    model_path: str,
    output_path: str,
    input_size: int = 1024,
    simplify_model: bool = True,
    opset_version: int = 11
) -> str:
    """
    Convert YOLO model to ONNX format.
    
    Args:
        model_path: Path to the PyTorch model file (.pt)
        output_path: Path to save the ONNX model (.onnx)
        input_size: Input image size for the model
        simplify_model: Whether to simplify the ONNX model
        opset_version: ONNX opset version
        
    Returns:
        Path to the converted ONNX model
    """
    logger.info(f"Loading PyTorch model from: {model_path}")
    
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Export to ONNX
    logger.info(f"Converting to ONNX format...")
    onnx_path = model.export(
        format='onnx',
        imgsz=input_size,
        opset=opset_version,
        simplify=simplify_model,
        dynamic=False,  # Use fixed input size for better optimization
        verbose=True
    )
    
    # Move to desired output path if different
    if onnx_path != output_path:
        import shutil
        shutil.move(onnx_path, output_path)
        onnx_path = output_path
    
    logger.info(f"ONNX model saved to: {onnx_path}")
    
    # Verify the ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
    except Exception as e:
        logger.warning(f"ONNX model validation failed: {e}")
    
    return onnx_path

def get_model_info(onnx_path: str):
    """Get information about the ONNX model."""
    try:
        model = onnx.load(onnx_path)
        
        # Get input/output info
        input_info = []
        output_info = []
        
        for input_tensor in model.graph.input:
            input_info.append({
                'name': input_tensor.name,
                'type': input_tensor.type.tensor_type.elem_type,
                'shape': [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            })
        
        for output_tensor in model.graph.output:
            output_info.append({
                'name': output_tensor.name,
                'type': output_tensor.type.tensor_type.elem_type,
                'shape': [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            })
        
        logger.info("Model Information:")
        logger.info(f"  Inputs: {input_info}")
        logger.info(f"  Outputs: {output_info}")
        
        # Get model size
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        logger.info(f"  Model size: {model_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX format")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to the PyTorch model file (.pt)"
    )
    parser.add_argument(
        "--output-path", 
        type=str,
        help="Path to save the ONNX model (.onnx). Default: same as input with .onnx extension"
    )
    parser.add_argument(
        "--input-size", 
        type=int, 
        default=1024,
        help="Input image size (default: 1024)"
    )
    parser.add_argument(
        "--no-simplify", 
        action="store_true",
        help="Skip model simplification"
    )
    parser.add_argument(
        "--opset-version", 
        type=int, 
        default=11,
        help="ONNX opset version (default: 11)"
    )
    
    args = parser.parse_args()
    
    # Validate input model
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Set output path
    if args.output_path is None:
        output_path = str(Path(args.model_path).with_suffix('.onnx'))
    else:
        output_path = args.output_path
    
    try:
        # Convert model
        onnx_path = convert_yolo_to_onnx(
            model_path=args.model_path,
            output_path=output_path,
            input_size=args.input_size,
            simplify_model=not args.no_simplify,
            opset_version=args.opset_version
        )
        
        # Get model information
        get_model_info(onnx_path)
        
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
