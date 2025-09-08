"""
Lightweight Document Layout Detection and Processing

This package provides a clean, lightweight implementation for document layout detection
using ONNX models and merging with PyMuPDF content extraction.
"""

from .detection import ONNXLayoutDetector
from .merging import PdfStyleCVMixLayoutExtractor
from .schemas import LayoutElement, LayoutExtractionResult, BoundingBox, ElementType

__version__ = "0.4.0"
__all__ = [
    "ONNXLayoutDetector",
    "PdfStyleCVMixLayoutExtractor", 
    "LayoutElement",
    "LayoutExtractionResult",
    "BoundingBox",
    "ElementType"
]
