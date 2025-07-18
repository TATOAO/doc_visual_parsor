"""
Computer Vision-based Document Layout Detection Module

This module provides a CV-based implementation of document layout detection using
the DocLayout-YOLO model. It supports both single images and multi-page PDFs.
"""

import logging
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
from PIL import Image
import torch
import io
import tempfile

try:
    import fitz  # PyMuPDF for PDF support
except ImportError:
    fitz = None

try:
    from doclayout_yolo import YOLOv10
except ImportError:
    try:
        # Fallback to ultralytics if doclayout_yolo is not available
        from ultralytics import YOLO as YOLOv10
    except ImportError:
        raise ImportError("Please install doclayout-yolo: pip install doclayout-yolo")

from ..base.base_layout_extractor import BaseLayoutExtractor
from doc_chunking.schemas.layout_schemas import (
    LayoutExtractionResult, 
    LayoutElement, 
    BoundingBox,
    ElementType
)
from .download_model import download_model, get_model_cache_dir
from ..utils.layout_processing import sort_elements_by_position, filter_redundant_boxes

logger = logging.getLogger(__name__)

# Mapping from DocLayout-YOLO class IDs to our standardized ElementType
# {0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption', 5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'}
DOCLAYOUT_CLASS_MAPPING = {
    0: ElementType.TITLE,
    1: ElementType.PLAIN_TEXT,
    2: ElementType.ABANDON,
    3: ElementType.FIGURE,
    4: ElementType.FIGURE_CAPTION,
    5: ElementType.TABLE,
    6: ElementType.TABLE_CAPTION,
    7: ElementType.TABLE_FOOTNOTE,
    8: ElementType.ISOLATE_FORMULA,
    9: ElementType.FORMULA_CAPTION
}

# Type alias for input data
InputDataType = Union[str, Path, bytes, io.BytesIO, io.BufferedReader, Any]

class CVLayoutDetector(BaseLayoutExtractor):
    """
    Computer Vision-based Document Layout Detector using DocLayout-YOLO.
    
    This class provides CV-based layout detection for both single images and multi-page PDFs.
    It inherits from BaseLayoutExtractor and implements the required abstract methods.
    """
    
    def __init__(self, 
                 model_name: str = "docstructbench",
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 confidence_threshold: float = 0.25,
                 image_size: int = 1024,
                 pdf_dpi: int = 150,  # Reduced default DPI for better scaling
                 **kwargs):
        """
        Initialize the CV-based layout detector.
        
        Args:
            model_name: Name of the model to use ('docstructbench', 'd4la', 'doclaynet')
            model_path: Custom path to model file (overrides model_name)
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence for detections
            image_size: Input image size for the model
            pdf_dpi: DPI resolution for PDF to image conversion
            **kwargs: Additional parameters
        """
        super().__init__(confidence_threshold=confidence_threshold, device=device, **kwargs)
        
        self.model_name = model_name
        self.image_size = image_size
        self.pdf_dpi = pdf_dpi
        self.model = None
        
        # Check PDF support
        if fitz is None:
            logger.warning("PyMuPDF not available. PDF support disabled.")
        
        # Determine model path
        if model_path:
            self.model_path = model_path
        else:
            logger.info(f"Will download/load model: {model_name}")
            self.model_path = None  # Will be set during initialization
    
    def _initialize_detector(self) -> None:
        """Initialize the detector by loading the YOLO model."""
        try:
            # Determine model path
            if self.model_path is None:
                # First try to find local model file
                current_dir = Path(__file__).parent
                local_model_path = current_dir / "model_parameters" / "docstructbench_doclayout_yolo_docstructbench_imgsz1024.pt"
                
                if local_model_path.exists():
                    logger.info(f"Using local model file: {local_model_path}")
                    self.model_path = str(local_model_path)
                else:
                    # Fall back to downloading model
                    logger.info(f"Local model not found, downloading model: {self.model_name}")
                    self.model_path = download_model(self.model_name)
            
            # Setup device
            self.device = self._setup_device(self.device)
            
            # Load model
            logger.info(f"Loading model from: {self.model_path}")
            self.model = YOLOv10(self.model_path, task="detect")
            logger.info("CV-based layout detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CV detector: {str(e)}")
            raise
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return device
    
    def _detect_layout(self, 
                      input_data: Any,
                      confidence_threshold: Optional[float] = None,
                      image_size: Optional[int] = None,
                      **kwargs) -> LayoutExtractionResult:
        """
        Core CV-based layout detection method.
        
        Args:
            input_data: Input data (image file path, numpy array, PIL Image, or PDF file path)
            confidence_threshold: Override default confidence threshold
            image_size: Override default image size
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected elements
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialization failed.")
        
        # Use instance defaults if not specified
        conf_thresh = confidence_threshold or self.confidence_threshold
        img_size = image_size or self.image_size
        
        # Handle different input types
        if self._is_pdf_input(input_data):
            return self._detect_layout_pdf(input_data, conf_thresh, img_size, **kwargs)
        else:
            return self._detect_layout_image(input_data, conf_thresh, img_size, **kwargs)
    
    def _is_pdf_input(self, input_data: Any) -> bool:
        """Check if input is a PDF file."""
        if isinstance(input_data, (str, Path)):
            return str(input_data).lower().endswith('.pdf')
        return False
    
    def _pdf_page_to_image(self, doc: fitz.Document, page_num: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Convert a PDF page to image for CV analysis.
        
        Args:
            doc: PyMuPDF Document object
            page_num: Page number (0-indexed)
            
        Returns:
            Tuple of (page image as numpy array, scale factor) or (None, 1.0) if failed
        """
        try:
            page = doc[page_num]
            
            # Calculate zoom factor based on desired DPI
            # Default PDF DPI is 72, so zoom = desired_dpi / 72
            zoom = self.pdf_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            img = Image.open(io.BytesIO(img_data))
            return np.array(img), 1.0 / zoom  # Return image and inverse scale factor
            
        except Exception as e:
            logger.warning(f"Could not convert page {page_num + 1} to image: {e}")
            return None, 1.0
    
    def _load_pdf_document(self, input_data: InputDataType) -> fitz.Document:
        """
        Load PDF document from various input types.
        
        Args:
            input_data: Input data in various formats
            
        Returns:
            PyMuPDF Document object
        """
        if isinstance(input_data, (str, Path)):
            return fitz.open(str(input_data))
        elif isinstance(input_data, bytes):
            return fitz.open(stream=input_data, filetype="pdf")
        elif hasattr(input_data, 'read'):
            # File-like object
            content = input_data.read()
            return fitz.open(stream=content, filetype="pdf")
        elif hasattr(input_data, 'getvalue'):
            # BytesIO or similar
            content = input_data.getvalue()
            return fitz.open(stream=content, filetype="pdf")
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    def _detect_layout_pdf(self, 
                          pdf_input: InputDataType,
                          confidence_threshold: float,
                          image_size: int,
                          **kwargs) -> LayoutExtractionResult:
        """
        Detect layout for PDF files by converting each page to image.
        
        Args:
            pdf_input: PDF file path or file object
            confidence_threshold: Confidence threshold
            image_size: Image size for detection
            **kwargs: Additional parameters
            
        Returns:
            LayoutExtractionResult with elements from all pages
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF not available. Cannot process PDF files.")
        
        try:
            # Load PDF document using the unified method
            doc = self._load_pdf_document(pdf_input)
            all_elements = []
            element_id = 0
            
            for page_num in range(doc.page_count):
                logger.info(f"Processing PDF page {page_num + 1}/{doc.page_count}")
                
                # Convert page to image
                page_image, scale_factor = self._pdf_page_to_image(doc, page_num)
                if page_image is None:
                    logger.warning(f"Could not convert page {page_num + 1} to image")
                    continue
                
                # Detect layout on the page image
                page_result = self._detect_layout_image(page_image, confidence_threshold, image_size, **kwargs)
                
                # Scale coordinates back to PDF space
                for element in page_result.elements:
                    if element.bbox:
                        element.bbox.x1 *= scale_factor
                        element.bbox.y1 *= scale_factor
                        element.bbox.x2 *= scale_factor
                        element.bbox.y2 *= scale_factor
                    
                    element.id = element_id
                    element.metadata = element.metadata or {}

                    element.metadata['page_number'] = page_num + 1
                    element.metadata['source_type'] = 'pdf_page'
                    element.metadata['scale_factor'] = scale_factor
                    all_elements.append(element)
                    element_id += 1
            
            # Create metadata
            metadata = {
                'detection_method': 'cv_yolo_pdf',
                'total_elements': len(all_elements),
                'document_type': 'pdf',
                'page_count': doc.page_count,
                'pdf_dpi': self.pdf_dpi
            }
            
            doc.close()
            
            return LayoutExtractionResult(elements=all_elements, metadata=metadata)
            
        except Exception as e:
            logger.error(f"PDF CV detection failed: {str(e)}")
            raise
    
    def _detect_layout_image(self, 
                           input_data: Any,
                           confidence_threshold: float,
                           image_size: int,
                           **kwargs) -> LayoutExtractionResult:
        """
        Core CV-based layout detection method for single images.
        
        Args:
            input_data: Input image (file path, numpy array, or PIL Image)
            confidence_threshold: Confidence threshold
            image_size: Image size for detection
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected elements
        """
        # Convert image to proper format
        if isinstance(input_data, (str, Path)):
            image_path = str(input_data)
        else:
            # Handle numpy array or PIL Image
            image_path = input_data
        
        try:
            # Perform prediction
            results = self.model.predict(
                image_path,
                imgsz=image_size,
                conf=confidence_threshold,
                device=self.device,
                verbose=False
            )

            # Extract results
            if len(results) == 0:
                return LayoutExtractionResult([])
            
            result = results[0]
            
            # Convert to our standardized format
            elements = []
            element_id = 0
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes_tensor = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                classes_tensor = result.boxes.cls.cpu().numpy().astype(int)
                conf_tensor = result.boxes.conf.cpu().numpy()
                
                for box, cls_id, conf in zip(boxes_tensor, classes_tensor, conf_tensor):
                    # Map class ID to our ElementType
                    element_type = DOCLAYOUT_CLASS_MAPPING.get(cls_id, ElementType.UNKNOWN)
                    
                    # Create bounding box
                    bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                    
                    # Create layout element
                    element = LayoutElement(
                        id=element_id,
                        element_type=element_type,
                        confidence=float(conf),
                        bbox=bbox,
                        metadata={
                            'model_class_id': int(cls_id),
                            'detection_method': 'cv_yolo',
                            'source_type': 'single_image'
                        }
                    )
                    
                    elements.append(element)
                    element_id += 1
            
            # Filter out redundant boxes
            elements = filter_redundant_boxes(elements)
            
            # Sort elements in natural reading order
            elements = sort_elements_by_position(elements)
            
            return LayoutExtractionResult(elements=elements)
            
        except Exception as e:
            logger.error(f"CV detection failed: {str(e)}")
            raise
    
    def detect_pdf_page(self, 
                       pdf_input: InputDataType, 
                       page_num: int,
                       **kwargs) -> LayoutExtractionResult:
        """
        Detect layout for a specific PDF page.
        
        Args:
            pdf_input: Path to PDF file or file object
            page_num: Page number (1-indexed)
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult for the specified page
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF not available. Cannot process PDF files.")
        
        try:
            doc = self._load_pdf_document(pdf_input)
            
            if page_num < 1 or page_num > doc.page_count:
                raise ValueError(f"Page number {page_num} out of range (1-{doc.page_count})")
            
            # Convert page to image (convert to 0-indexed)
            page_image, scale_factor = self._pdf_page_to_image(doc, page_num - 1)
            doc.close()
            
            if page_image is None:
                raise RuntimeError(f"Could not convert page {page_num} to image")
            
            # Detect layout
            result = self._detect_layout_image(page_image, 
                                             self.confidence_threshold, 
                                             self.image_size, 
                                             **kwargs)
            
            # Add page information
            for element in result.elements:
                element.metadata = element.metadata or {}
                element.metadata['page_number'] = page_num
                element.metadata['source_type'] = 'pdf_page'
            
            return result
            
        except Exception as e:
            logger.error(f"PDF page detection failed: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats."""
        formats = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            'PIL.Image', 'numpy.ndarray', 'file_path'
        ]
        
        if fitz is not None:
            formats.append('.pdf')
        
        return formats
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the CV detector."""
        return {
            'detector_type': 'cv_based',
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'image_size': self.image_size,
            'pdf_dpi': self.pdf_dpi,
            'pdf_support': fitz is not None,
            'supported_formats': self.get_supported_formats(),
            'class_mapping': {k: v.value for k, v in DOCLAYOUT_CLASS_MAPPING.items()}
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate if input data is supported by CV detector."""
        if isinstance(input_data, (str, Path)):
            # Check if file exists and has valid extension
            path = Path(input_data)
            if not path.exists():
                return False
            
            # Check supported formats
            ext = path.suffix.lower()
            if ext == '.pdf':
                return fitz is not None
            else:
                return ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        elif isinstance(input_data, bytes):
            # Assume it's a PDF if it's bytes
            return fitz is not None
        elif hasattr(input_data, 'read') or hasattr(input_data, 'getvalue'):
            # File-like object, assume PDF
            return fitz is not None
        elif isinstance(input_data, Image.Image):
            return True
        elif isinstance(input_data, np.ndarray):
            return len(input_data.shape) >= 2  # At least 2D array
        else:
            return False
    
    def visualize(self, 
                  input_data: Any,
                  result: Optional[LayoutExtractionResult] = None,
                  save_path: Optional[str] = None,
                  line_width: int = 3,
                  font_size: int = 16,
                  **kwargs) -> np.ndarray:
        """
        Visualize detection results on the image.
        
        Args:
            input_data: Input image
            result: Detection result (if None, will run detection)
            save_path: Path to save the annotated image
            line_width: Line width for bounding boxes
            font_size: Font size for labels
            **kwargs: Additional visualization parameters
            
        Returns:
            Annotated image as numpy array
        """
        # Run detection if result not provided
        if result is None:
            result = self.detect(input_data)
        
        # Load image for visualization
        if isinstance(input_data, (str, Path)):
            img = cv2.imread(str(input_data))
            if img is None:
                raise ValueError(f"Could not load image from {input_data}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(input_data, Image.Image):
            img = np.array(input_data)
        else:
            img = input_data.copy()
        
        # Colors for different element types (RGB)
        colors = {
            ElementType.PLAIN_TEXT: (0, 255, 0),    # Green
            ElementType.TITLE: (255, 0, 0),         # Red
            ElementType.HEADING: (255, 0, 0),       # Red
            ElementType.FIGURE: (0, 0, 255),        # Blue
            ElementType.FIGURE_CAPTION: (0, 100, 255), # Light Blue
            ElementType.TABLE: (255, 255, 0),       # Yellow
            ElementType.TABLE_CAPTION: (255, 200, 0), # Orange
            ElementType.HEADER: (255, 0, 255),      # Magenta
            ElementType.FOOTER: (255, 0, 255),      # Magenta
            ElementType.REFERENCE: (128, 128, 128), # Gray
            ElementType.EQUATION: (255, 128, 0),    # Orange
            ElementType.LIST: (0, 255, 128),        # Light Green
            ElementType.PARAGRAPH: (0, 255, 0),     # Green
            ElementType.UNKNOWN: (128, 128, 128)    # Gray
        }
        
        # Draw bounding boxes and labels
        for element in result.get_elements():
            if element.bbox is None:
                continue
                
            bbox = element.bbox
            color = colors.get(element.element_type, (128, 128, 128))
            label = f"{element.element_type.value}: {element.confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(img, 
                         (int(bbox.x1), int(bbox.y1)),
                         (int(bbox.x2), int(bbox.y2)),
                         color, line_width)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20, 2)[0]
            cv2.rectangle(img,
                         (int(bbox.x1), int(bbox.y1) - label_size[1] - 10),
                         (int(bbox.x1) + label_size[0], int(bbox.y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(img, label,
                       (int(bbox.x1), int(bbox.y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size / 20, (255, 255, 255), 2)
        
        # Save if requ!.gitignoreested
        if save_path:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            logger.info(f"Annotated image saved to: {save_path}")
        
        return img


# python -m models.layout_detection.visual_detection.cv_detector
if __name__ == "__main__":
    detector = CVLayoutDetector()
    detector._initialize_detector()

    pdf_path = "tests/test_data/1-1 买卖合同（通用版）.pdf"
    import fitz

    result = detector._detect_layout(pdf_path)

    import json

    json.dump(result.model_dump(), open("test_image_result_cv.json", "w"), indent=2, ensure_ascii=False)

    from ..utils.visualiza_layout_elements import visualize_pdf_layout
    visualize_pdf_layout(pdf_path, result, "visualized_cv_only.pdf")