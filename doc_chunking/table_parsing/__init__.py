"""
Lightweight Table Parsing Module

This module provides table detection, structure recognition, and content extraction
using ONNX models without PyTorch dependencies.
"""

from .table_detector import ONNXTableDetector
from .table_structure import ONNXTableStructureRecognizer
from .table_ocr import TableOCRProcessor
from .table_parser import TableParser
from .schemas import TableElement, TableStructure, TableCell, TableRow, TableColumn

__all__ = [
    "ONNXTableDetector",
    "ONNXTableStructureRecognizer", 
    "TableOCRProcessor",
    "TableParser",
    "TableElement",
    "TableStructure",
    "TableCell",
    "TableRow",
    "TableColumn"
]
