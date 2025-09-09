"""
Convert DocLayout-YOLO PyTorch models to ONNX format.

This script converts the downloaded PyTorch models to ONNX format for use with
the lightweight ONNX-based detection system.
"""

import os
import logging
import torch
import torch.onnx
from pathlib import Path
import argparse
from typing import Optional

logger = logging.getLogger(__name__)

def convert_pytorch_to_onnx(
    pytorch_model_path: str,
    output_path: Optional[str] = None,
    input_size: int = 1024,
    batch_size: int = 1,
    device: str = "cpu"
) -> str:
    """
    Convert a PyTorch DocLayout-YOLO model to ONNX format.
    
    Args:
        pytorch_model_path: Path to the PyTorch model file (.pt)
        output_path: Output path for ONNX model (optional)
        input_size: Input image size (default: 1024)
        batch_size: Batch size for conversion (default: 1)
        device: Device to use for conversion ('cpu' or 'cuda')
        
    Returns:
        Path to the converted ONNX model
    """
    pytorch_path = Path(pytorch_model_path)
    if not pytorch_path.exists():
        raise FileNotFoundError(f"PyTorch model not found: {pytorch_model_path}")
    
    # Set output path if not provided
    if output_path is None:
        output_path = pytorch_path.parent / f"{pytorch_path.stem}.onnx"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Converting PyTorch model: {pytorch_path}")
    logger.info(f"Output ONNX model: {output_path}")
    logger.info(f"Input size: {input_size}x{input_size}")
    logger.info(f"Device: {device}")
    
    try:
        # Load the PyTorch model
        logger.info("Loading PyTorch model...")
        model = torch.load(pytorch_model_path, map_location=device, weights_only=False)
        
        # Handle different model formats
        if isinstance(model, dict):
            # If it's a state dict, we need to reconstruct the model
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                # This is a state dict, we need the model architecture
                logger.warning("Model appears to be a state dict. You may need to provide the model architecture.")
                # For YOLO models, we'll try to use the model directly
                model = model
            else:
                # Try to use the model directly
                model = model
        
        # Set model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Convert model to float32 if it's in half precision
        if hasattr(model, 'half') and next(model.parameters()).dtype == torch.float16:
            logger.info("Converting model from half precision to float32...")
            model = model.float()
        
        # Create dummy input tensor
        dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device, dtype=torch.float32)
        
        # Convert to ONNX
        logger.info("Converting to ONNX format...")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,  # Use ONNX opset 11 for better compatibility
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Successfully converted model to ONNX: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to convert model: {str(e)}")
        raise

def convert_doclayout_yolo_specific(
    pytorch_model_path: str,
    output_path: Optional[str] = None,
    input_size: int = 1024
) -> str:
    """
    Convert DocLayout-YOLO model with specific handling for YOLO architecture.
    
    Args:
        pytorch_model_path: Path to the PyTorch model file
        output_path: Output path for ONNX model
        input_size: Input image size
        
    Returns:
        Path to the converted ONNX model
    """
    pytorch_path = Path(pytorch_model_path)
    if output_path is None:
        output_path = pytorch_path.parent / f"{pytorch_path.stem}.onnx"
    else:
        output_path = Path(output_path)
    
    logger.info(f"Converting DocLayout-YOLO model: {pytorch_path}")
    
    try:
        # Load the model
        model = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        
        # Handle YOLO model structure
        if isinstance(model, dict):
            if 'model' in model:
                yolo_model = model['model']
            else:
                # Try to extract the model from the checkpoint
                yolo_model = model
        else:
            yolo_model = model
        
        # Set to evaluation mode
        if hasattr(yolo_model, 'eval'):
            yolo_model.eval()
        
        # Convert model to float32 if it's in half precision
        if hasattr(yolo_model, 'half') and next(yolo_model.parameters()).dtype == torch.float16:
            logger.info("Converting YOLO model from half precision to float32...")
            yolo_model = yolo_model.float()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
        
        # Export to ONNX with YOLO-specific settings
        torch.onnx.export(
            yolo_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Successfully converted DocLayout-YOLO to ONNX: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to convert DocLayout-YOLO model: {str(e)}")
        logger.info("Trying alternative conversion method...")
        
        # Try the generic conversion as fallback
        return convert_pytorch_to_onnx(pytorch_model_path, str(output_path), input_size)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX format")
    parser.add_argument("--input", "-i", required=True,
                       help="Path to input PyTorch model (.pt file)")
    parser.add_argument("--output", "-o",
                       help="Path to output ONNX model (.onnx file)")
    parser.add_argument("--input-size", type=int, default=1024,
                       help="Input image size (default: 1024)")
    parser.add_argument("--device", default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to use for conversion (default: cpu)")
    parser.add_argument("--model-type", default="doclayout_yolo",
                       choices=["doclayout_yolo", "generic"],
                       help="Type of model to convert (default: doclayout_yolo)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        if args.model_type == "doclayout_yolo":
            output_path = convert_doclayout_yolo_specific(
                args.input,
                args.output,
                args.input_size
            )
        else:
            output_path = convert_pytorch_to_onnx(
                args.input,
                args.output,
                args.input_size,
                device=args.device
            )
        
        print(f"Conversion successful! ONNX model saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return 1
    
    return 0

def exam_output(onnx_model_path: str, pt_model_path: str):
    """Exam the output of the ONNX model."""

    # onnx model
    import onnxruntime
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    output_onnx = ort_session.run(None, {input_name: dummy_input.numpy()})
    
    # original model
    from doclayout_yolo import YOLOv10
    model = YOLOv10(pt_model_path)

    output_pt = model.predict(dummy_input)

    # compare the output
    print(f"Output shape: {output_onnx.shape}")
    print(f"Output shape: {output_pt.shape}")

    # compare the output
    print(f"Output difference: {output_onnx - output_pt}")


# python scripts/convert_to_onnx.py --input model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.pt --output model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx --model-type doclayout_yolo
# python -m scripts.convert_to_onnx
if __name__ == "__main__":
    # main()

    exam_output(
        "model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx", 
        "model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.pt"
    )