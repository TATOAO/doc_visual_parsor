"""
Convert DocLayout-YOLO PyTorch models to ONNX format.

Require torch and DocLayout-YOLO.

pip install doclayout-yolo
pip install onnx

This script converts the downloaded PyTorch models to ONNX format for use with
the lightweight ONNX-based detection system.
"""
import os
import argparse
from doclayout_yolo import YOLOv10


def export_to_onnx(model_path, output_path, imgsz=1024, opset=11, simplify=True, dynamic=False):
    """
    Export a DocLayout-YOLO model to ONNX format.
    
    Args:
        model_path (str): Path to the PyTorch model file (.pt)
        output_path (str): Output path for the ONNX model (.onnx)
        imgsz (int): Input image size for the model
        opset (int): ONNX opset version
        simplify (bool): Whether to simplify the ONNX model
        dynamic (bool): Whether to use dynamic axes for batch size and image dimensions
    """
    print(f"Loading model from: {model_path}")
    
    # Load the model
    model = YOLOv10(model_path, task='detect')
    
    print(f"Model loaded successfully. Exporting to ONNX format...")
    print(f"Input image size: {imgsz}")
    print(f"ONNX opset version: {opset}")
    print(f"Simplify model: {simplify}")
    print(f"Dynamic axes: {dynamic}")
    
    # Export to ONNX
    try:
        exported_model = model.export(
            format='onnx',
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            verbose=True
        )
        
        print(f"✅ Model successfully exported to: {exported_model}")
        
        # Verify the exported model
        if os.path.exists(exported_model):
            file_size = os.path.getsize(exported_model) / (1024 * 1024)  # MB
            print(f"📁 Exported ONNX model size: {file_size:.2f} MB")
            
            # Test loading the ONNX model
            try:
                import onnx
                onnx_model = onnx.load(exported_model)
                print(f"✅ ONNX model validation passed")
                print(f"📊 Model inputs: {[input.name for input in onnx_model.graph.input]}")
                print(f"📊 Model outputs: {[output.name for output in onnx_model.graph.output]}")
            except ImportError:
                print("⚠️  ONNX package not available for validation")
            except Exception as e:
                print(f"⚠️  ONNX model validation failed: {e}")
        
        return exported_model
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


# python -m scripts.convert_to_onnx --model model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.pt --output model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx
def main():
    parser = argparse.ArgumentParser(description='Export DocLayout-YOLO model to ONNX format')
    parser.add_argument('--model', required=True, type=str, 
                       help='Path to the PyTorch model file (.pt)')
    parser.add_argument('--output', default=None, type=str,
                       help='Output path for the ONNX model (.onnx). If not specified, will use model name with .onnx extension')
    parser.add_argument('--imgsz', default=1024, type=int,
                       help='Input image size (default: 1024)')
    parser.add_argument('--opset', default=11, type=int,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify the ONNX model (default: True)')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false',
                       help='Disable ONNX model simplification')
    parser.add_argument('--dynamic', action='store_true', default=False,
                       help='Use dynamic axes for batch size and image dimensions (default: False)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    # Set output path if not provided
    if args.output is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"{model_name}.onnx"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        exported_path = export_to_onnx(
            model_path=args.model,
            output_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=args.dynamic
        )
        
        print(f"\n🎉 Export completed successfully!")
        print(f"📁 ONNX model saved to: {exported_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Export failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
