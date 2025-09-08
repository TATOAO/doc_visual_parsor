"""
Example usage of the lightweight document layout detection system.

This example demonstrates how to use the new clean API for document layout detection
and content extraction.
"""

import logging
from pathlib import Path
from .detection import ONNXLayoutDetector
from .merging import PdfStyleCVMixLayoutExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_onnx_detection():
    """Example of using ONNX layout detection only."""
    logger.info("=== ONNX Layout Detection Example ===")
    
    # Initialize detector
    model_path = "path/to/your/onnx/model.onnx"  # Replace with actual model path
    detector = ONNXLayoutDetector(
        model_path=model_path,
        confidence_threshold=0.25,
        image_size=1024
    )
    
    # Detect layout in a PDF
    pdf_path = "path/to/your/document.pdf"  # Replace with actual PDF path
    
    try:
        result = detector.detect_layout(pdf_path)
        
        logger.info(f"Detected {len(result.elements)} layout elements")
        
        # Print summary of detected elements
        for element in result.elements[:5]:  # Show first 5 elements
            logger.info(f"Element {element.id}: {element.element_type} "
                       f"(confidence: {element.confidence:.3f})")
            if element.text:
                logger.info(f"  Text: {element.text[:50]}...")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Detection failed: {e}")


def example_hybrid_extraction():
    """Example of using hybrid CV + PDF extraction."""
    logger.info("=== Hybrid CV + PDF Extraction Example ===")
    
    # Initialize hybrid extractor
    model_path = "path/to/your/onnx/model.onnx"  # Replace with actual model path
    extractor = PdfStyleCVMixLayoutExtractor(
        model_path=model_path,
        cv_confidence_threshold=0.25,
        cv_image_size=1024
    )
    
    # Extract layout and content from PDF
    pdf_path = "path/to/your/document.pdf"  # Replace with actual PDF path
    
    try:
        result = extractor.detect_layout(pdf_path)
        
        logger.info(f"Extracted {len(result.elements)} enriched elements")
        logger.info(f"Metadata: {result.metadata}")
        
        # Print summary of enriched elements
        for element in result.elements[:5]:  # Show first 5 elements
            logger.info(f"Element {element.id}: {element.element_type}")
            if element.text:
                logger.info(f"  Text: {element.text[:100]}...")
            if element.style and element.style.runs:
                logger.info(f"  Runs: {len(element.style.runs)} text runs with formatting")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")


def example_image_detection():
    """Example of detecting layout in a single image."""
    logger.info("=== Image Layout Detection Example ===")
    
    # Initialize detector
    model_path = "path/to/your/onnx/model.onnx"  # Replace with actual model path
    detector = ONNXLayoutDetector(
        model_path=model_path,
        confidence_threshold=0.3
    )
    
    # Detect layout in an image
    image_path = "path/to/your/image.png"  # Replace with actual image path
    
    try:
        result = detector.detect_layout(image_path)
        
        logger.info(f"Detected {len(result.elements)} layout elements in image")
        
        # Print detected elements
        for element in result.elements:
            logger.info(f"Element {element.id}: {element.element_type} "
                       f"at ({element.bbox.x1:.1f}, {element.bbox.y1:.1f}, "
                       f"{element.bbox.x2:.1f}, {element.bbox.y2:.1f})")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"Detection failed: {e}")


if __name__ == "__main__":
    # Run examples (uncomment the ones you want to test)
    
    # example_onnx_detection()
    # example_hybrid_extraction()
    # example_image_detection()
    
    logger.info("Examples completed. Uncomment the example functions to run them.")
