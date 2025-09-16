"""
Table structure recognition module.

This module provides table structure recognition using ONNX models
to identify rows, columns, and cell boundaries in tables.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path

from .schemas import TableStructure, TableRow, TableColumn, TableCell, TableType, CellType
from .table_ocr import TableOCRProcessor

logger = logging.getLogger(__name__)


class ONNXTableStructureRecognizer:
    """
    ONNX-based table structure recognizer.
    
    This class uses ONNX models to recognize table structure including
    rows, columns, and cell boundaries without PyTorch dependencies.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize the table structure recognizer.
        
        Args:
            model_path: Path to ONNX model file
            confidence_threshold: Minimum confidence for structure detection
            device: Device to use ('auto', 'cpu', 'cuda')
            **kwargs: Additional parameters
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_size = 1024  # Default input size
        
        # Initialize model if path provided
        if self.model_path and os.path.exists(self.model_path):
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ONNX model."""
        try:
            # Setup providers
            providers = self._get_onnx_providers()
            
            # Load ONNX model
            logger.info(f"Loading table structure model from: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:  # NCHW format
                self.input_size = input_shape[2]  # Height
            
            logger.info(f"Table structure model loaded successfully")
            logger.info(f"Input size: {self.input_size}x{self.input_size}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize table structure model: {str(e)}")
            raise
    
    def _get_onnx_providers(self) -> List[str]:
        """Get ONNX Runtime providers based on device."""
        available_providers = ort.get_available_providers()
        
        if self.device == "cuda" and 'CUDAExecutionProvider' in available_providers:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for table structure recognition.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (preprocessed_image, scale_x, scale_y)
        """
        original_height, original_width = image.shape[:2]
        
        # Resize image while maintaining aspect ratio
        scale = min(self.input_size / original_width, self.input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_height, :new_width] = resized
        
        # Convert BGR to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        input_tensor = np.transpose(padded, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor, scale, scale
    
    def recognize_structure(self, 
                           table_image: np.ndarray,
                           ocr_results: Optional[List[Dict[str, Any]]] = None) -> TableStructure:
        """
        Recognize table structure from an image.
        
        Args:
            table_image: Table region image
            ocr_results: Optional OCR results for text content
            
        Returns:
            TableStructure object with recognized structure
        """
        if self.session is None:
            # Fallback to rule-based structure recognition
            return self._rule_based_structure_recognition(table_image, ocr_results)
        
        # Preprocess image
        input_tensor, scale_x, scale_y = self._preprocess_image(table_image)
        
        try:
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Postprocess outputs to get structure
            structure = self._postprocess_structure(outputs, table_image, scale_x, scale_y, ocr_results)
            
            return structure
            
        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            # Fallback to rule-based recognition
            return self._rule_based_structure_recognition(table_image, ocr_results)
    
    def _postprocess_structure(self, 
                              outputs: List[np.ndarray],
                              table_image: np.ndarray,
                              scale_x: float,
                              scale_y: float,
                              ocr_results: Optional[List[Dict[str, Any]]]) -> TableStructure:
        """
        Postprocess model outputs to extract table structure.
        
        Args:
            outputs: Raw model outputs
            table_image: Original table image
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
            ocr_results: OCR results for text content
            
        Returns:
            TableStructure object
        """
        # This is a simplified implementation
        # In practice, you would need to implement the specific postprocessing
        # based on your ONNX model's output format
        
        # For now, fall back to rule-based recognition
        return self._rule_based_structure_recognition(table_image, ocr_results)
    
    def _rule_based_structure_recognition(self, 
                                         table_image: np.ndarray,
                                         ocr_results: Optional[List[Dict[str, Any]]]) -> TableStructure:
        """
        Rule-based table structure recognition as fallback.
        
        Args:
            table_image: Table region image
            ocr_results: OCR results for text content
            
        Returns:
            TableStructure object
        """
        # Convert to grayscale
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_lines = self._detect_horizontal_lines(gray)
        vertical_lines = self._detect_vertical_lines(gray)
        
        # Create grid from detected lines
        rows, cols = self._create_grid_from_lines(horizontal_lines, vertical_lines, table_image.shape)
        
        # Create table structure
        table_rows = []
        table_columns = []
        
        # Create rows
        for i in range(rows):
            cells = []
            for j in range(cols):
                # Calculate cell bounding box
                cell_bbox = self._calculate_cell_bbox(i, j, rows, cols, table_image.shape)
                
                # Get text content for this cell
                cell_text = self._extract_cell_text(cell_bbox, ocr_results)
                
                # Determine cell type
                cell_type = CellType.HEADER if i == 0 else CellType.DATA
                
                cell = TableCell(
                    row=i,
                    col=j,
                    text=cell_text,
                    cell_type=cell_type,
                    bbox=cell_bbox
                )
                cells.append(cell)
            
            table_row = TableRow(
                index=i,
                cells=cells,
                is_header=(i == 0)
            )
            table_rows.append(table_row)
        
        # Create columns
        for j in range(cols):
            table_column = TableColumn(
                index=j,
                is_header=(j == 0)
            )
            table_columns.append(table_column)
        
        # Determine table type
        table_type = self._determine_table_type(horizontal_lines, vertical_lines)
        
        return TableStructure(
            rows=table_rows,
            columns=table_columns,
            row_count=rows,
            col_count=cols,
            has_header=True,  # Assume first row is header
            table_type=table_type
        )
    
    def _detect_horizontal_lines(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect horizontal lines in the image."""
        # Apply horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50:  # Filter short lines
                lines.append((x, y, x + w, y + h))
        
        return lines
    
    def _detect_vertical_lines(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect vertical lines in the image."""
        # Apply vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > 50:  # Filter short lines
                lines.append((x, y, x + w, y + h))
        
        return lines
    
    def _create_grid_from_lines(self, 
                               horizontal_lines: List[Tuple[int, int, int, int]],
                               vertical_lines: List[Tuple[int, int, int, int]],
                               image_shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Create grid from detected lines.
        
        Args:
            horizontal_lines: List of horizontal line coordinates
            vertical_lines: List of vertical line coordinates
            image_shape: Image shape (height, width, channels)
            
        Returns:
            Tuple of (rows, columns)
        """
        # Sort lines by position
        horizontal_lines.sort(key=lambda x: x[1])  # Sort by y
        vertical_lines.sort(key=lambda x: x[0])    # Sort by x
        
        # Count unique row and column positions
        row_positions = set()
        col_positions = set()
        
        for line in horizontal_lines:
            row_positions.add(line[1])  # y position
        
        for line in vertical_lines:
            col_positions.add(line[0])  # x position
        
        # Add image boundaries
        row_positions.add(0)
        row_positions.add(image_shape[0])
        col_positions.add(0)
        col_positions.add(image_shape[1])
        
        # Convert to sorted lists
        row_positions = sorted(list(row_positions))
        col_positions = sorted(list(col_positions))
        
        # Calculate grid dimensions
        rows = max(1, len(row_positions) - 1)
        cols = max(1, len(col_positions) - 1)
        
        return rows, cols
    
    def _calculate_cell_bbox(self, 
                            row: int, 
                            col: int, 
                            total_rows: int, 
                            total_cols: int,
                            image_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for a cell."""
        height, width = image_shape[:2]
        
        # Calculate cell boundaries
        cell_width = width / total_cols
        cell_height = height / total_rows
        
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = (col + 1) * cell_width
        y2 = (row + 1) * cell_height
        
        return (x1, y1, x2, y2)
    
    def _extract_cell_text(self, 
                          cell_bbox: Tuple[float, float, float, float],
                          ocr_results: Optional[List[Dict[str, Any]]]) -> str:
        """Extract text content for a cell from OCR results."""
        if not ocr_results:
            return ""
        
        x1, y1, x2, y2 = cell_bbox
        cell_texts = []
        
        for result in ocr_results:
            bbox = result['bbox']
            text_x1, text_y1, text_x2, text_y2 = bbox
            
            # Check if OCR result is within cell bounds
            if (text_x1 >= x1 and text_y1 >= y1 and 
                text_x2 <= x2 and text_y2 <= y2):
                cell_texts.append(result['text'])
        
        return " ".join(cell_texts)
    
    def _determine_table_type(self, 
                             horizontal_lines: List[Tuple[int, int, int, int]],
                             vertical_lines: List[Tuple[int, int, int, int]]) -> TableType:
        """Determine table type based on detected lines."""
        if len(horizontal_lines) > 0 and len(vertical_lines) > 0:
            return TableType.WIRED
        elif len(horizontal_lines) > 0 or len(vertical_lines) > 0:
            return TableType.MIXED
        else:
            return TableType.WIRELESS
    
    def get_recognizer_info(self) -> Dict[str, Any]:
        """Get information about the structure recognizer."""
        return {
            'recognizer_type': 'onnx_table_structure',
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.input_size,
            'model_loaded': self.session is not None,
            'providers': self.session.get_providers() if self.session else []
        }
