"""
Test script for DocLayout-YOLO integration

This script tests the layout detection functionality without requiring actual image files.
"""

import sys
import logging
from pathlib import Path

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_download():
    """Test model download functionality"""
    logger.info("Testing model download...")
    
    try:
        from models.layout_detection import download_model, list_available_models
        
        # List available models
        logger.info("Available models:")
        list_available_models()
        
        # Test download (this will download the model if not already present)
        model_path = download_model("docstructbench")
        logger.info(f"Model downloaded to: {model_path}")
        
        return True
    except Exception as e:
        logger.error(f"Model download test failed: {str(e)}")
        return False

def test_detector_initialization():
    """Test detector initialization"""
    logger.info("Testing detector initialization...")
    
    try:
        from models.layout_detection import DocLayoutDetector
        
        # Initialize detector
        detector = DocLayoutDetector(model_name="docstructbench")
        
        # Get model info
        info = detector.get_model_info()
        logger.info(f"Detector initialized successfully:")
        logger.info(f"  Model: {info['model_name']}")
        logger.info(f"  Device: {info['device']}")
        logger.info(f"  Confidence threshold: {info['confidence_threshold']}")
        logger.info(f"  Image size: {info['image_size']}")
        
        return True
    except Exception as e:
        logger.error(f"Detector initialization test failed: {str(e)}")
        return False

def test_api_imports():
    """Test API imports"""
    logger.info("Testing API imports...")
    
    try:
        # Test importing the API server with layout detection
        from backend.api_server import app
        logger.info("API server imports successful")
        
        # Check if our new endpoints are available
        routes = [route.path for route in app.routes]
        
        expected_routes = ["/api/detect-layout", "/api/visualize-layout"]
        for route in expected_routes:
            if route in routes:
                logger.info(f"✓ Route {route} is available")
            else:
                logger.warning(f"✗ Route {route} is missing")
        
        return True
    except Exception as e:
        logger.error(f"API imports test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting DocLayout-YOLO integration tests...")
    
    tests = [
        ("Model Download", test_model_download),
        ("Detector Initialization", test_detector_initialization), 
        ("API Imports", test_api_imports)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            logger.error(f"Test {test_name} encountered an error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! DocLayout-YOLO integration is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 