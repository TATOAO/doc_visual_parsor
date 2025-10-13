"""
Lightweight Document Layout Detection and Processing

This package provides a clean, lightweight implementation for document layout detection
using ONNX models and merging with PyMuPDF content extraction.
"""

from .onnx_layout_detector import ONNXDocLayoutYOLO
from .merging import PdfStyleCVMixLayoutExtractor
from .schemas import LayoutElement, LayoutExtractionResult, BoundingBox, ElementType
from .core.processors.simplified_processor import SimplifiedProcessor

__version__ = "0.4.0"
__all__ = [
    "ONNXLayoutDetector",
    "PdfStyleCVMixLayoutExtractor", 
    "LayoutElement",
    "LayoutExtractionResult",
    "BoundingBox",
    "ElementType"
]

# python -m doc_chunking.__init__
if __name__ == "__main__":
    import asyncio
    result = asyncio.run(SimplifiedProcessor().process("/Users/tatoao_mini/Downloads/劳动合同(1).docx"))
    print(result)
