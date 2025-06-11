"""
Demo script for the new modular layout detection architecture

This script demonstrates how to use the different detector types:
1. CV-based detector (for images)
2. Document-native detector (for .docx files)
3. Hybrid detector (combines multiple methods)
4. Factory pattern for automatic detector selection
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import List, Any

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.layout_detection import (
    LayoutDetectorFactory,
    DetectorType,
    CVLayoutDetector,
    DocumentLayoutDetector,
    HybridLayoutDetector,
    ElementType
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_cv_detector(image_path: str):
    """Demonstrate CV-based layout detection."""
    print("\n" + "="*50)
    print("CV-BASED LAYOUT DETECTION DEMO")
    print("="*50)
    
    try:
        # Create CV detector
        detector = CVLayoutDetector(
            model_name="docstructbench",
            confidence_threshold=0.25
        )
        
        # Get detector info
        info = detector.get_detector_info()
        print(f"Detector Type: {info['detector_type']}")
        print(f"Supported Formats: {info['supported_formats']}")
        
        # Perform detection
        print(f"\nAnalyzing image: {image_path}")
        result = detector.detect(image_path)
        
        # Display results
        print(f"\nDetected {len(result.elements)} elements:")
        for element in result.get_elements():
            bbox = element.bbox
            print(f"  {element.element_type.value} (confidence: {element.confidence:.3f}) "
                  f"at [{bbox.x1:.0f}, {bbox.y1:.0f}, {bbox.x2:.0f}, {bbox.y2:.0f}]")
        
        # Show statistics
        stats = result.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total elements: {stats['total_elements']}")
        for elem_type, count in stats['element_types'].items():
            print(f"  {elem_type}: {count}")
        
        return result
        
    except Exception as e:
        print(f"CV detection failed: {str(e)}")
        return None

def demo_document_detector(docx_path: str):
    """Demonstrate document-native layout detection."""
    print("\n" + "="*50)
    print("DOCUMENT-NATIVE LAYOUT DETECTION DEMO")
    print("="*50)
    
    try:
        # Create document detector
        detector = DocumentLayoutDetector(
            confidence_threshold=0.8,
            extract_text=True,
            analyze_styles=True
        )
        
        # Get detector info
        info = detector.get_detector_info()
        print(f"Detector Type: {info['detector_type']}")
        print(f"Supported Formats: {info['supported_formats']}")
        print(f"Capabilities: {info['capabilities']}")
        
        # Perform detection
        print(f"\nAnalyzing document: {docx_path}")
        result = detector.detect(docx_path)
        
        # Display results
        print(f"\nDetected {len(result.elements)} elements:")
        for element in result.get_elements():
            text_preview = element.text[:50] + "..." if element.text and len(element.text) > 50 else element.text
            print(f"  {element.element_type.value} (confidence: {element.confidence:.3f})")
            if text_preview:
                print(f"    Text: {text_preview}")
            if element.metadata:
                print(f"    Style: {element.metadata.get('style_name', 'N/A')}")
        
        # Show statistics
        stats = result.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total elements: {stats['total_elements']}")
        for elem_type, count in stats['element_types'].items():
            print(f"  {elem_type}: {count}")
        
        return result
        
    except Exception as e:
        print(f"Document detection failed: {str(e)}")
        return None

def demo_hybrid_detector(input_path: str):
    """Demonstrate hybrid layout detection."""
    print("\n" + "="*50)
    print("HYBRID LAYOUT DETECTION DEMO")
    print("="*50)
    
    try:
        # Create hybrid detector
        detector = HybridLayoutDetector(
            detector_types=[DetectorType.CV_BASED, DetectorType.DOCUMENT_NATIVE],
            combination_strategy="merge",
            confidence_threshold=0.5
        )
        
        # Get detector info
        info = detector.get_detector_info()
        print(f"Detector Type: {info['detector_type']}")
        print(f"Combination Strategy: {info['combination_strategy']}")
        print(f"Number of Sub-detectors: {len(info['sub_detectors'])}")
        
        # Perform detection
        print(f"\nAnalyzing with hybrid approach: {input_path}")
        result = detector.detect(input_path)
        
        # Display results
        print(f"\nDetected {len(result.elements)} elements:")
        for element in result.get_elements():
            source = element.metadata.get('source_detector', 'Unknown') if element.metadata else 'Unknown'
            print(f"  {element.element_type.value} (confidence: {element.confidence:.3f}) "
                  f"from {source}")
        
        # Show statistics
        stats = result.get_statistics()
        print(f"\nStatistics:")
        print(f"  Total elements: {stats['total_elements']}")
        for elem_type, count in stats['element_types'].items():
            print(f"  {elem_type}: {count}")
        
        return result
        
    except Exception as e:
        print(f"Hybrid detection failed: {str(e)}")
        return None

def demo_factory_auto_selection(input_paths: List[str]):
    """Demonstrate automatic detector selection using factory."""
    print("\n" + "="*50)
    print("FACTORY AUTO-SELECTION DEMO")
    print("="*50)
    
    for input_path in input_paths:
        print(f"\nProcessing: {input_path}")
        
        try:
            # Auto-select best detector
            detector = LayoutDetectorFactory.auto_detect_best_detector(
                input_path,
                confidence_threshold=0.3
            )
            
            print(f"Auto-selected detector: {type(detector).__name__}")
            
            # Perform detection
            result = detector.detect(input_path)
            print(f"Found {len(result.elements)} elements")
            
        except Exception as e:
            print(f"Auto-detection failed for {input_path}: {str(e)}")

def demo_available_detectors():
    """Show information about available detectors."""
    print("\n" + "="*50)
    print("AVAILABLE DETECTORS")
    print("="*50)
    
    detectors = LayoutDetectorFactory.get_available_detectors()
    
    for detector_name, info in detectors.items():
        print(f"\n{detector_name.upper()}:")
        print(f"  Type: {info['detector_type']}")
        print(f"  Supported formats: {info['supported_formats']}")
        if 'capabilities' in info:
            print(f"  Capabilities: {info['capabilities']}")

def main():
    parser = argparse.ArgumentParser(description="Modular Layout Detection Demo")
    parser.add_argument("--image", type=str, help="Path to image file for CV detection")
    parser.add_argument("--docx", type=str, help="Path to DOCX file for document detection")
    parser.add_argument("--hybrid", type=str, help="Path to file for hybrid detection")
    parser.add_argument("--auto", type=str, nargs="+", help="Path(s) to files for auto-selection demo")
    parser.add_argument("--show-detectors", action="store_true", help="Show available detectors")
    
    args = parser.parse_args()
    
    # Show available detectors
    if args.show_detectors:
        demo_available_detectors()
    
    # Run CV detector demo
    if args.image:
        if Path(args.image).exists():
            demo_cv_detector(args.image)
        else:
            print(f"Image file not found: {args.image}")
    
    # Run document detector demo
    if args.docx:
        if Path(args.docx).exists():
            demo_document_detector(args.docx)
        else:
            print(f"DOCX file not found: {args.docx}")
    
    # Run hybrid detector demo
    if args.hybrid:
        if Path(args.hybrid).exists():
            demo_hybrid_detector(args.hybrid)
        else:
            print(f"File not found: {args.hybrid}")
    
    # Run auto-selection demo
    if args.auto:
        existing_files = [f for f in args.auto if Path(f).exists()]
        if existing_files:
            demo_factory_auto_selection(existing_files)
        else:
            print("No valid files found for auto-selection demo")
    
    # If no arguments provided, show help
    if not any([args.image, args.docx, args.hybrid, args.auto, args.show_detectors]):
        parser.print_help()
        print("\nExample usage:")
        print("  python demo_modular_detection.py --show-detectors")
        print("  python demo_modular_detection.py --image path/to/image.jpg")
        print("  python demo_modular_detection.py --docx path/to/document.docx")
        print("  python demo_modular_detection.py --hybrid path/to/file")
        print("  python demo_modular_detection.py --auto path/to/file1 path/to/file2")

if __name__ == "__main__":
    main() 