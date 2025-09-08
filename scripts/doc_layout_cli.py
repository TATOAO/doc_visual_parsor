#!/usr/bin/env python3
"""
Lightweight Document Layout Detection Command-Line Tool

This is a minimal command-line tool for document layout detection using ONNX Runtime
instead of PyTorch, resulting in much smaller memory footprint and faster startup.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Minimal dependencies - only import what we need
try:
    import numpy as np
    from PIL import Image
    import cv2
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install numpy pillow opencv-python")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Missing ONNX Runtime. Please install: pip install onnxruntime")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not available. PDF support disabled.")

# Import our lightweight detector
from doc_chunking.layout_detection.visual_detection.onnx_detector import ONNXLayoutDetector
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentLayoutCLI:
    """Lightweight command-line interface for document layout detection."""
    
    def __init__(self, model_path: str, confidence: float = 0.25, image_size: int = 1024):
        """
        Initialize the CLI tool.
        
        Args:
            model_path: Path to the ONNX model file
            confidence: Confidence threshold for detections
            image_size: Input image size for the model
        """
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.detector = None
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
    
    def initialize(self):
        """Initialize the detector."""
        logger.info("Initializing ONNX-based layout detector...")
        self.detector = ONNXLayoutDetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence,
            image_size=self.image_size
        )
        self.detector._initialize_detector()
        logger.info("Detector initialized successfully")
    
    def detect_file(self, input_path: str, output_path: Optional[str] = None) -> dict:
        """
        Detect layout for a single file.
        
        Args:
            input_path: Path to input file (image or PDF)
            output_path: Optional path to save results JSON
            
        Returns:
            Detection results as dictionary
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Processing file: {input_path}")
        
        # Run detection
        result = self.detector._detect_layout(input_path)
        
        # Convert to dictionary
        result_dict = result.model_dump()
        
        # Save results if output path specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        
        return result_dict
    
    def detect_batch(self, input_dir: str, output_dir: str, file_extensions: List[str] = None) -> List[dict]:
        """
        Detect layout for multiple files in a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save results
            file_extensions: List of file extensions to process
            
        Returns:
            List of detection results
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all files with supported extensions
        files_to_process = []
        for ext in file_extensions:
            files_to_process.extend(input_path.glob(f"*{ext}"))
            files_to_process.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not files_to_process:
            logger.warning(f"No supported files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        results = []
        for i, file_path in enumerate(files_to_process, 1):
            try:
                logger.info(f"Processing {i}/{len(files_to_process)}: {file_path.name}")
                
                # Generate output filename
                output_file = output_path / f"{file_path.stem}_layout.json"
                
                # Detect layout
                result = self.detect_file(str(file_path), str(output_file))
                results.append({
                    'file': str(file_path),
                    'output': str(output_file),
                    'elements_count': len(result.get('elements', [])),
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results.append({
                    'file': str(file_path),
                    'error': str(e),
                    'success': False
                })
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.detector is None:
            raise RuntimeError("Detector not initialized")
        
        return self.detector.get_detector_info()

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight Document Layout Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect layout for a single PDF
  python doc_layout_cli.py --model model.onnx --input document.pdf --output results.json
  
  # Process all PDFs in a directory
  python doc_layout_cli.py --model model.onnx --batch input_dir/ output_dir/
  
  # Get model information
  python doc_layout_cli.py --model model.onnx --info
        """
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Path to ONNX model file"
    )
    
    parser.add_argument(
        "--input", 
        help="Input file path (for single file processing)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output JSON file path (for single file processing)"
    )
    
    parser.add_argument(
        "--batch", 
        nargs=2,
        metavar=("INPUT_DIR", "OUTPUT_DIR"),
        help="Process all files in input directory and save to output directory"
    )
    
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--image-size", 
        type=int, 
        default=1024,
        help="Input image size (default: 1024)"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show model information and exit"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize CLI tool
        cli = DocumentLayoutCLI(
            model_path=args.model,
            confidence=args.confidence,
            image_size=args.image_size
        )
        cli.initialize()
        
        # Show model info if requested
        if args.info:
            info = cli.get_model_info()
            print("Model Information:")
            print(json.dumps(info, indent=2))
            return
        
        # Process files
        if args.batch:
            input_dir, output_dir = args.batch
            logger.info(f"Batch processing: {input_dir} -> {output_dir}")
            results = cli.detect_batch(input_dir, output_dir)
            
            # Print summary
            successful = sum(1 for r in results if r.get('success', False))
            failed = len(results) - successful
            print(f"\nBatch processing completed:")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            
        elif args.input:
            if not args.output:
                # Generate default output filename
                input_path = Path(args.input)
                args.output = str(input_path.with_suffix('.json'))
            
            result = cli.detect_file(args.input, args.output)
            elements_count = len(result.get('elements', []))
            print(f"Detection completed: {elements_count} elements found")
            
        else:
            parser.error("Either --input or --batch must be specified")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
