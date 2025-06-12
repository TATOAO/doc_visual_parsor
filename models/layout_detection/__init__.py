# Acknowledgement: https://github.com/opendatalab/DocLayout-YOLO
# DocLayout-YOLO is a model for detecting the layout of a document.
# It is a YOLO-based model that is trained on a dataset of document images.
# It is used to detect the layout of a document, such as the presence of a table, a list, or a text block.

# New modular architecture
from .base.base_detector import (
    BaseLayoutDetector,
    BaseSectionDetector
)

from .visual_detection.cv_detector import CVLayoutDetector
from .style_detection.docx_detector import SectionLayoutDetector
# from .detector_factory import (
#     LayoutDetectorFactory,
#     HybridLayoutDetector,
#     DetectorType
# )

# Backward compatibility
from .visual_detection.cv_detector import DocLayoutDetector  # Alias for CVLayoutDetector
from .visual_detection.download_model import download_model, list_available_models, MODELS

__all__ = [
    # Base classes
    'BaseLayoutDetector',
    'LayoutDetectionResult', 
    'LayoutElement',
    'BoundingBox',
    'ElementType',
    
    # Detector implementations
    'CVLayoutDetector',
    'DocumentLayoutDetector',
    'HybridLayoutDetector',
    
    # Factory and utilities
    'LayoutDetectorFactory',
    'DetectorType',
    
    # Backward compatibility
    'DocLayoutDetector',
    'download_model',
    'list_available_models', 
    'MODELS'
]



