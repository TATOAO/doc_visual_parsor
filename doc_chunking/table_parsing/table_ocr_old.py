"""
Table OCR processing module.

This module provides OCR functionality specifically for table text extraction
using lightweight OCR engines without PyTorch dependencies.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TableOCRProcessor:
    """
    OCR processor specifically designed for table text extraction.
    
    This class provides OCR functionality for extracting text from table regions
    using lightweight OCR engines without PyTorch dependencies.
    """
    
    def __init__(self, 
                 ocr_engine: str = "auto",
                 languages: List[str] = ["en"],
                 **kwargs):
        """
        Initialize the table OCR processor.
        
        Args:
            ocr_engine: OCR engine to use ('auto', 'easyocr', 'tesseract')
            languages: List of languages for OCR
            **kwargs: Additional parameters
        """
        self.ocr_engine = ocr_engine
        self.languages = languages
        self.engine = None
        
        # Initialize OCR engine
        self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """Initialize the OCR engine."""
        if self.ocr_engine == "auto":
            # Try to use EasyOCR first, then Tesseract
            if EASYOCR_AVAILABLE:
                self.ocr_engine = "easyocr"
            elif TESSERACT_AVAILABLE:
                self.ocr_engine = "tesseract"
            else:
                raise RuntimeError("No OCR engine available. Please install easyocr or pytesseract.")
        
        if self.ocr_engine == "easyocr" and EASYOCR_AVAILABLE:
            try:
                self.engine = easyocr.Reader(self.languages, gpu=False)
                logger.info(f"Initialized EasyOCR with languages: {self.languages}")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
        elif self.ocr_engine == "tesseract" and TESSERACT_AVAILABLE:
            self.engine = "tesseract"
            logger.info(f"Initialized Tesseract with languages: {self.languages}")
        else:
            raise RuntimeError(f"OCR engine '{self.ocr_engine}' not available or not installed.")
    
    def extract_text_from_image(self, 
                               image: np.ndarray,
                               bbox: Optional[Tuple[float, float, float, float]] = None) -> List[Dict[str, Any]]:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Input image as numpy array
            bbox: Optional bounding box to crop image (x1, y1, x2, y2)
            
        Returns:
            List of OCR results with text, confidence, and bounding box
        """
        # Crop image if bbox provided
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            image = image[y1:y2, x1:x2]
        
        # Preprocess image for better OCR
        processed_image = self._preprocess_image(image)
        
        if self.ocr_engine == "easyocr":
            return self._extract_with_easyocr(processed_image)
        elif self.ocr_engine == "tesseract":
            return self._extract_with_tesseract(processed_image)
        else:
            raise RuntimeError(f"Unsupported OCR engine: {self.ocr_engine}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_with_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR."""
        try:
            results = self.engine.readtext(image)
            
            ocr_results = []
            for detection in results:
                bbox, text, confidence = detection
                
                # Convert bbox format from EasyOCR to our format
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                if len(bbox) == 4:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    ocr_results.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'bbox_points': bbox
                    })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def _extract_with_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract."""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                lang='+'.join(self.languages)
            )
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                # Skip empty text or low confidence
                if not text or confidence < 30:
                    continue
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                ocr_results.append({
                    'text': text,
                    'confidence': confidence / 100.0,  # Convert to 0-1 range
                    'bbox': [x, y, x + w, y + h],
                    'bbox_points': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def extract_table_text(self, 
                          table_image: np.ndarray,
                          table_bbox: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
        """
        Extract text from a table image with specific processing for tables.
        
        Args:
            table_image: Table region image
            table_bbox: Table bounding box in original image coordinates
            
        Returns:
            List of OCR results with text, confidence, and bounding box
        """
        # Extract text from the table image
        ocr_results = self.extract_text_from_image(table_image)
        
        # Adjust bounding boxes to original image coordinates
        x1_table, y1_table, x2_table, y2_table = table_bbox
        table_width = x2_table - x1_table
        table_height = y2_table - y1_table
        
        # Scale factor for adjusting coordinates
        scale_x = table_width / table_image.shape[1] if table_image.shape[1] > 0 else 1.0
        scale_y = table_height / table_image.shape[0] if table_image.shape[0] > 0 else 1.0
        
        adjusted_results = []
        for result in ocr_results:
            # Adjust bounding box coordinates
            bbox = result['bbox']
            adjusted_bbox = [
                bbox[0] * scale_x + x1_table,
                bbox[1] * scale_y + y1_table,
                bbox[2] * scale_x + x1_table,
                bbox[3] * scale_y + y1_table
            ]
            
            adjusted_result = result.copy()
            adjusted_result['bbox'] = adjusted_bbox
            adjusted_results.append(adjusted_result)
        
        return adjusted_results
    
    def group_text_by_lines(self, 
                           ocr_results: List[Dict[str, Any]], 
                           line_threshold: float = 10.0) -> List[List[Dict[str, Any]]]:
        """
        Group OCR results by text lines.
        
        Args:
            ocr_results: List of OCR results
            line_threshold: Vertical distance threshold for grouping lines
            
        Returns:
            List of text lines, each containing OCR results
        """
        if not ocr_results:
            return []
        
        # Sort by y-coordinate (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x: x['bbox'][1])
        
        lines = []
        current_line = [sorted_results[0]]
        
        for i in range(1, len(sorted_results)):
            current_bbox = sorted_results[i]['bbox']
            prev_bbox = sorted_results[i-1]['bbox']
            
            # Check if this text is on the same line
            y_diff = abs(current_bbox[1] - prev_bbox[1])
            
            if y_diff <= line_threshold:
                current_line.append(sorted_results[i])
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [sorted_results[i]]
        
        # Add the last line
        lines.append(current_line)
        
        # Sort each line by x-coordinate (left to right)
        for line in lines:
            line.sort(key=lambda x: x['bbox'][0])
        
        return lines
    
    def get_ocr_info(self) -> Dict[str, Any]:
        """Get information about the OCR processor."""
        return {
            'ocr_engine': self.ocr_engine,
            'languages': self.languages,
            'easyocr_available': EASYOCR_AVAILABLE,
            'tesseract_available': TESSERACT_AVAILABLE,
            'engine_initialized': self.engine is not None
        }
