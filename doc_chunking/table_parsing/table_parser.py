"""
Main table parser that integrates table detection, OCR, and structure recognition.

This module provides a unified interface for table parsing without PyTorch dependencies.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path

from .table_detector import ONNXTableDetector
from .table_ocr import TableOCRProcessor
from .table_structure import ONNXTableStructureRecognizer
from .schemas import TableElement, TableParsingResult, TableType

logger = logging.getLogger(__name__)


class TableParser:
    """
    Main table parser that integrates all table processing components.
    
    This class provides a unified interface for:
    - Table detection
    - OCR text extraction
    - Structure recognition
    - Content formatting
    """
    
    def __init__(self,
                 table_detector: Optional[ONNXTableDetector] = None,
                 ocr_processor: Optional[TableOCRProcessor] = None,
                 structure_recognizer: Optional[ONNXTableStructureRecognizer] = None,
                 **kwargs):
        """
        Initialize the table parser.
        
        Args:
            table_detector: Table detector instance
            ocr_processor: OCR processor instance
            structure_recognizer: Structure recognizer instance
            **kwargs: Additional parameters
        """
        self.table_detector = table_detector
        self.ocr_processor = ocr_processor
        self.structure_recognizer = structure_recognizer
        
        # Initialize default components if not provided
        if self.table_detector is None:
            self.table_detector = ONNXTableDetector(**kwargs)
        
        if self.ocr_processor is None:
            self.ocr_processor = TableOCRProcessor(**kwargs)
        
        if self.structure_recognizer is None:
            self.structure_recognizer = ONNXTableStructureRecognizer(**kwargs)
    
    def parse_tables(self, 
                    image: Union[str, np.ndarray, Path],
                    detect_tables: bool = True,
                    extract_text: bool = True,
                    recognize_structure: bool = True,
                    **kwargs) -> TableParsingResult:
        """
        Parse tables from an image.
        
        Args:
            image: Input image (file path, numpy array, or Path)
            detect_tables: Whether to detect table regions
            extract_text: Whether to extract text content
            recognize_structure: Whether to recognize table structure
            **kwargs: Additional parameters
            
        Returns:
            TableParsingResult with parsed tables
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img = image.copy()
        
        tables = []
        
        if detect_tables:
            # Detect table regions
            detected_tables = self.table_detector.detect_tables(img, **kwargs)
            logger.info(f"Detected {len(detected_tables)} table regions")
        else:
            # Assume entire image is a table
            h, w = img.shape[:2]
            detected_tables = [TableElement(
                id=0,
                bbox=(0, 0, w, h),
                confidence=1.0,
                metadata={'detection_method': 'full_image'}
            )]
        
        # Process each detected table
        for i, table in enumerate(detected_tables):
            try:
                # Crop table region
                table_image = self.table_detector.crop_table_region(img, table)
                
                # Extract text if requested
                ocr_results = None
                if extract_text:
                    ocr_results = self.ocr_processor.extract_table_text(
                        table_image, table.bbox
                    )
                    logger.info(f"Extracted text from {len(ocr_results)} regions in table {i}")
                
                # Recognize structure if requested
                if recognize_structure:
                    structure = self.structure_recognizer.recognize_structure(
                        table_image, ocr_results
                    )
                    table.structure = structure
                    
                    # Generate formatted content
                    table.html_content = structure.to_html()
                    table.markdown_content = structure.to_markdown()
                    table.csv_content = structure.to_csv()
                    
                    logger.info(f"Recognized structure for table {i}: {structure.row_count}x{structure.col_count}")
                
                # Update metadata
                table.metadata = table.metadata or {}
                table.metadata.update({
                    'ocr_results_count': len(ocr_results) if ocr_results else 0,
                    'has_structure': table.structure is not None,
                    'processing_completed': True
                })
                
                tables.append(table)
                
            except Exception as e:
                logger.error(f"Failed to process table {i}: {e}")
                # Add table with error information
                table.metadata = table.metadata or {}
                table.metadata.update({
                    'processing_error': str(e),
                    'processing_completed': False
                })
                tables.append(table)
        
        # Create result
        result = TableParsingResult(
            tables=tables,
            metadata={
                'total_tables': len(tables),
                'tables_with_structure': len([t for t in tables if t.has_structure]),
                'detection_enabled': detect_tables,
                'text_extraction_enabled': extract_text,
                'structure_recognition_enabled': recognize_structure
            }
        )
        
        logger.info(f"Table parsing completed: {result.table_count} tables processed")
        return result
    
    def parse_single_table(self, 
                          table_image: np.ndarray,
                          table_bbox: Optional[Tuple[float, float, float, float]] = None,
                          **kwargs) -> TableElement:
        """
        Parse a single table from an image.
        
        Args:
            table_image: Table region image
            table_bbox: Optional bounding box in original image coordinates
            **kwargs: Additional parameters
            
        Returns:
            TableElement with parsed content
        """
        # Create table element
        if table_bbox is None:
            h, w = table_image.shape[:2]
            table_bbox = (0, 0, w, h)
        
        table = TableElement(
            id=0,
            bbox=table_bbox,
            confidence=1.0,
            metadata={'detection_method': 'single_table'}
        )
        
        try:
            # Extract text
            ocr_results = self.ocr_processor.extract_table_text(table_image, table_bbox)
            
            # Recognize structure
            structure = self.structure_recognizer.recognize_structure(table_image, ocr_results)
            table.structure = structure
            
            # Generate formatted content
            table.html_content = structure.to_html()
            table.markdown_content = structure.to_markdown()
            table.csv_content = structure.to_csv()
            
            # Update metadata
            table.metadata.update({
                'ocr_results_count': len(ocr_results),
                'has_structure': True,
                'processing_completed': True
            })
            
        except Exception as e:
            logger.error(f"Failed to parse single table: {e}")
            table.metadata.update({
                'processing_error': str(e),
                'processing_completed': False
            })
        
        return table
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the table parser."""
        return {
            'parser_type': 'integrated_table_parser',
            'table_detector_info': self.table_detector.get_detector_info() if self.table_detector else None,
            'ocr_processor_info': self.ocr_processor.get_ocr_info() if self.ocr_processor else None,
            'structure_recognizer_info': self.structure_recognizer.get_recognizer_info() if self.structure_recognizer else None
        }
    
    def visualize_tables(self, 
                        image: np.ndarray, 
                        tables: List[TableElement],
                        output_path: Optional[str] = None,
                        show_structure: bool = True) -> np.ndarray:
        """
        Visualize detected tables on the image.
        
        Args:
            image: Original image
            tables: List of table elements
            output_path: Optional path to save visualization
            show_structure: Whether to show table structure
            
        Returns:
            Image with table visualizations
        """
        vis_image = image.copy()
        
        # Define colors for different elements
        table_color = (0, 255, 0)  # Green for table boundaries
        cell_color = (255, 0, 0)   # Red for cell boundaries
        text_color = (0, 0, 255)   # Blue for text
        
        for i, table in enumerate(tables):
            # Draw table boundary
            x1, y1, x2, y2 = map(int, table.bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), table_color, 2)
            
            # Add table label
            label = f"Table {i+1} ({table.confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, table_color, 2)
            
            # Draw table structure if available
            if show_structure and table.structure:
                self._draw_table_structure(vis_image, table, cell_color)
        
        # Save visualization if path provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
            logger.info(f"Table visualization saved to: {output_path}")
        
        return vis_image
    
    def _draw_table_structure(self, 
                             image: np.ndarray, 
                             table: TableElement, 
                             cell_color: Tuple[int, int, int]):
        """Draw table structure on the image."""
        if not table.structure:
            return
        
        x1_table, y1_table, x2_table, y2_table = map(int, table.bbox)
        
        for row in table.structure.rows:
            for cell in row.cells:
                if cell.bbox:
                    # Scale cell coordinates to image coordinates
                    cell_x1 = int(cell.bbox[0] + x1_table)
                    cell_y1 = int(cell.bbox[1] + y1_table)
                    cell_x2 = int(cell.bbox[2] + x1_table)
                    cell_y2 = int(cell.bbox[3] + y1_table)
                    
                    # Draw cell boundary
                    cv2.rectangle(image, (cell_x1, cell_y1), (cell_x2, cell_y2), cell_color, 1)
                    
                    # Add cell text if available
                    if cell.text:
                        text = cell.text[:20] + "..." if len(cell.text) > 20 else cell.text
                        cv2.putText(image, text, (cell_x1+2, cell_y1+15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
