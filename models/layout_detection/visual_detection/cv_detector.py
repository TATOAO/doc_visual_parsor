"""
Computer Vision-based Document Layout Detection Module

This module provides a CV-based implementation of document layout detection using
the DocLayout-YOLO model. It inherits from the BaseLayoutDetector.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
from PIL import Image
import torch

try:
    from doclayout_yolo import YOLOv10
except ImportError:
    try:
        # Fallback to ultralytics if doclayout_yolo is not available
        from ultralytics import YOLO as YOLOv10
    except ImportError:
        raise ImportError("Please install doclayout-yolo: pip install doclayout-yolo")

from ..base.base_layout_extractor import BaseSectionDetector
from models.schemas.layout_schemas import (
    LayoutExtractionResult, 
    LayoutElement, 
    BoundingBox,
    ElementType
)
from .download_model import download_model, get_model_cache_dir

logger = logging.getLogger(__name__)

# Mapping from DocLayout-YOLO class IDs to our standardized ElementType
DOCLAYOUT_CLASS_MAPPING = {
    0: ElementType.TEXT,
    1: ElementType.TITLE, 
    2: ElementType.FIGURE,
    3: ElementType.FIGURE_CAPTION,
    4: ElementType.TABLE,
    5: ElementType.TABLE_CAPTION,
    6: ElementType.HEADER,
    7: ElementType.FOOTER,
    8: ElementType.REFERENCE,
    9: ElementType.EQUATION
}

class CVLayoutDetector(BaseSectionDetector):
    """
    Computer Vision-based Document Layout Detector using DocLayout-YOLO.
    
    This class provides CV-based layout detection for document images.
    It inherits from BaseLayoutDetector and implements the required abstract methods.
    """
    
    def __init__(self, 
                 model_name: str = "docstructbench",
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 confidence_threshold: float = 0.25,
                 image_size: int = 1024,
                 **kwargs):
        """
        Initialize the CV-based layout detector.
        
        Args:
            model_name: Name of the model to use ('docstructbench', 'd4la', 'doclaynet')
            model_path: Custom path to model file (overrides model_name)
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence for detections
            image_size: Input image size for the model
            **kwargs: Additional parameters
        """
        super().__init__(confidence_threshold=confidence_threshold, device=device, **kwargs)
        
        self.model_name = model_name
        self.image_size = image_size
        self.model = None
        
        # Determine model path
        if model_path:
            self.model_path = model_path
        else:
            logger.info(f"Will download/load model: {model_name}")
            self.model_path = None  # Will be set during initialization
    
    def _initialize_detector(self) -> None:
        """Initialize the detector by loading the YOLO model."""
        try:
            # Download model if needed
            if self.model_path is None:
                logger.info(f"Downloading/loading model: {self.model_name}")
                self.model_path = download_model(self.model_name)
            
            # Setup device
            self.device = self._setup_device(self.device)
            
            # Load model
            logger.info(f"Loading model from: {self.model_path}")
            self.model = YOLOv10(self.model_path)
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
            input_data: Input image (file path, numpy array, or PIL Image)
            confidence_threshold: Override default confidence threshold
            image_size: Override default image size
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutDetectionResult containing detected elements
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Initialization failed.")
        
        # Use instance defaults if not specified
        conf_thresh = confidence_threshold or self.confidence_threshold
        img_size = image_size or self.image_size
        
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
                imgsz=img_size,
                conf=conf_thresh,
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
                            'detection_method': 'cv_yolo'
                        }
                    )
                    
                    elements.append(element)
                    element_id += 1
            
            return LayoutExtractionResult(elements)
            
        except Exception as e:
            logger.error(f"CV detection failed: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats."""
        return [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            'PIL.Image', 'numpy.ndarray', 'file_path'
        ]
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the CV detector."""
        return {
            'detector_type': 'cv_based',
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'image_size': self.image_size,
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
            return path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
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
            ElementType.TEXT: (0, 255, 0),          # Green
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
        
        # Save if requested
        if save_path:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            logger.info(f"Annotated image saved to: {save_path}")
        
        return img

# Backward compatibility alias
DocLayoutDetector = CVLayoutDetector 