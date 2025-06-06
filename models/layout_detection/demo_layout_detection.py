"""
Demo script for DocLayout-YOLO Document Layout Detection

This script demonstrates how to use the DocLayoutDetector to analyze document layouts.
"""

import logging
import sys
from pathlib import Path
from PIL import Image
import argparse

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent))

from models.layout_detection import DocLayoutDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DocLayout-YOLO Demo")
    parser.add_argument("--image", type=str, required=True, 
                       help="Path to the input image")
    parser.add_argument("--output", type=str, 
                       help="Path to save the annotated output image")
    parser.add_argument("--model", type=str, default="docstructbench",
                       choices=["docstructbench", "d4la", "doclaynet"],
                       help="Model to use for detection")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--image-size", type=int, default=1024,
                       help="Input image size for the model")
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not Path(args.image).exists():
        logger.error(f"Input image not found: {args.image}")
        return
    
    try:
        # Initialize the detector
        logger.info("Initializing DocLayout-YOLO detector...")
        detector = DocLayoutDetector(
            model_name=args.model,
            device=args.device,
            confidence_threshold=args.confidence,
            image_size=args.image_size
        )
        
        # Print model info
        model_info = detector.get_model_info()
        logger.info(f"Model loaded: {model_info['model_name']}")
        logger.info(f"Device: {model_info['device']}")
        logger.info(f"Confidence threshold: {model_info['confidence_threshold']}")
        
        # Perform detection
        logger.info(f"Analyzing layout of: {args.image}")
        result = detector.detect(args.image)
        
        # Display results
        elements = result.get_elements()
        logger.info(f"Detected {len(elements)} layout elements:")
        
        for element in elements:
            bbox = element['bbox']
            logger.info(f"  {element['type']} (confidence: {element['confidence']:.3f}) "
                       f"at [{bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f}]")
        
        # Create visualization
        if args.output or len(elements) > 0:
            output_path = args.output or f"annotated_{Path(args.image).name}"
            logger.info(f"Creating annotated image: {output_path}")
            
            annotated_img = detector.visualize(
                args.image, 
                result, 
                save_path=output_path
            )
            
            logger.info(f"Annotated image saved to: {output_path}")
        
        # Print summary by element type
        type_counts = {}
        for element in elements:
            element_type = element['type']
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        if type_counts:
            logger.info("Summary by element type:")
            for element_type, count in sorted(type_counts.items()):
                logger.info(f"  {element_type}: {count}")
        
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 