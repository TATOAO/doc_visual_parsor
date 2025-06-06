"""
DocLayout-YOLO Document Layout Detection Module

This module provides a high-level interface for document layout detection using
the DocLayout-YOLO model. It can detect various document elements like text blocks,
tables, figures, lists, etc.
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

from models.layout_detection.download_model import download_model, get_model_cache_dir

logger = logging.getLogger(__name__)

# Document layout class names (based on DocStructBench)
LAYOUT_CLASSES = {
    0: "Text",
    1: "Title", 
    2: "Figure",
    3: "Figure Caption",
    4: "Table",
    5: "Table Caption",
    6: "Header",
    7: "Footer",
    8: "Reference",
    9: "Equation"
}

class LayoutDetectionResult:
    """Container for layout detection results."""
    
    def __init__(self, 
                 boxes: List[List[float]], 
                 classes: List[int], 
                 confidences: List[float],
                 class_names: Dict[int, str] = None):
        self.boxes = boxes  # List of [x1, y1, x2, y2] coordinates
        self.classes = classes  # List of class IDs
        self.confidences = confidences  # List of confidence scores
        self.class_names = class_names or LAYOUT_CLASSES
        
    def get_elements(self) -> List[Dict[str, Any]]:
        """Get detected elements as a list of dictionaries."""
        elements = []
        for i, (box, cls_id, conf) in enumerate(zip(self.boxes, self.classes, self.confidences)):
            elements.append({
                'id': i,
                'type': self.class_names.get(cls_id, f'Unknown_{cls_id}'),
                'class_id': cls_id,
                'confidence': conf,
                'bbox': {
                    'x1': box[0],
                    'y1': box[1], 
                    'x2': box[2],
                    'y2': box[3],
                    'width': box[2] - box[0],
                    'height': box[3] - box[1]
                }
            })
        return elements
    
    def filter_by_confidence(self, min_confidence: float = 0.5):
        """Filter results by minimum confidence threshold."""
        filtered_indices = [i for i, conf in enumerate(self.confidences) if conf >= min_confidence]
        
        return LayoutDetectionResult(
            boxes=[self.boxes[i] for i in filtered_indices],
            classes=[self.classes[i] for i in filtered_indices], 
            confidences=[self.confidences[i] for i in filtered_indices],
            class_names=self.class_names
        )
    
    def filter_by_type(self, element_types: List[str]):
        """Filter results by element types."""
        type_ids = [cls_id for cls_id, name in self.class_names.items() if name in element_types]
        filtered_indices = [i for i, cls_id in enumerate(self.classes) if cls_id in type_ids]
        
        return LayoutDetectionResult(
            boxes=[self.boxes[i] for i in filtered_indices],
            classes=[self.classes[i] for i in filtered_indices],
            confidences=[self.confidences[i] for i in filtered_indices], 
            class_names=self.class_names
        )

class DocLayoutDetector:
    """
    Document Layout Detector using DocLayout-YOLO.
    
    This class provides an easy-to-use interface for detecting document layout elements
    such as text blocks, tables, figures, etc.
    """
    
    def __init__(self, 
                 model_name: str = "docstructbench",
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 confidence_threshold: float = 0.25,
                 image_size: int = 1024):
        """
        Initialize the DocLayout detector.
        
        Args:
            model_name: Name of the model to use ('docstructbench', 'd4la', 'doclaynet')
            model_path: Custom path to model file (overrides model_name)
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence for detections
            image_size: Input image size for the model
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = None
        
        # Load model
        if model_path:
            self.model_path = model_path
        else:
            logger.info(f"Downloading/loading model: {model_name}")
            self.model_path = download_model(model_name)
            
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """Load the YOLO model."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = YOLOv10(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def detect(self, 
               image: Union[str, Path, np.ndarray, Image.Image],
               confidence_threshold: Optional[float] = None,
               image_size: Optional[int] = None) -> LayoutDetectionResult:
        """
        Detect layout elements in an image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            confidence_threshold: Override default confidence threshold
            image_size: Override default image size
            
        Returns:
            LayoutDetectionResult containing detected elements
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please check initialization.")
        
        # Use instance defaults if not specified
        conf_thresh = confidence_threshold or self.confidence_threshold
        img_size = image_size or self.image_size
        
        # Convert image to proper format
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            # Handle numpy array or PIL Image
            image_path = image
        
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
                return LayoutDetectionResult([], [], [])
            
            result = results[0]
            
            # Extract bounding boxes, classes, and confidences
            boxes = []
            classes = []
            confidences = []
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes_tensor = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                classes_tensor = result.boxes.cls.cpu().numpy().astype(int)
                conf_tensor = result.boxes.conf.cpu().numpy()
                
                boxes = boxes_tensor.tolist()
                classes = classes_tensor.tolist() 
                confidences = conf_tensor.tolist()
            
            return LayoutDetectionResult(boxes, classes, confidences, LAYOUT_CLASSES)
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise
    
    def detect_batch(self, 
                    images: List[Union[str, Path, np.ndarray, Image.Image]],
                    confidence_threshold: Optional[float] = None,
                    image_size: Optional[int] = None) -> List[LayoutDetectionResult]:
        """
        Detect layout elements in multiple images.
        
        Args:
            images: List of input images
            confidence_threshold: Override default confidence threshold  
            image_size: Override default image size
            
        Returns:
            List of LayoutDetectionResult objects
        """
        results = []
        for image in images:
            try:
                result = self.detect(image, confidence_threshold, image_size)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image {image}: {str(e)}")
                results.append(LayoutDetectionResult([], [], []))
        
        return results
    
    def visualize(self, 
                  image: Union[str, Path, np.ndarray, Image.Image],
                  result: Optional[LayoutDetectionResult] = None,
                  confidence_threshold: Optional[float] = None,
                  save_path: Optional[str] = None,
                  line_width: int = 3,
                  font_size: int = 16) -> np.ndarray:
        """
        Visualize detection results on the image.
        
        Args:
            image: Input image
            result: Detection result (if None, will run detection)
            confidence_threshold: Minimum confidence for visualization
            save_path: Path to save the annotated image
            line_width: Line width for bounding boxes
            font_size: Font size for labels
            
        Returns:
            Annotated image as numpy array
        """
        # Run detection if result not provided
        if result is None:
            result = self.detect(image, confidence_threshold)
        
        # Load image for visualization
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Draw bounding boxes and labels
        for element in result.get_elements():
            bbox = element['bbox']
            label = f"{element['type']}: {element['confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(img, 
                         (int(bbox['x1']), int(bbox['y1'])),
                         (int(bbox['x2']), int(bbox['y2'])),
                         (0, 255, 0), line_width)
            
            # Draw label
            cv2.putText(img, label,
                       (int(bbox['x1']), int(bbox['y1']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_size / 20, (0, 255, 0), 2)
        
        # Save if requested
        if save_path:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_bgr)
            logger.info(f"Annotated image saved to: {save_path}")
        
        return img

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'image_size': self.image_size,
            'class_names': LAYOUT_CLASSES
        } 


# python -m models.layout_detection.detector
if __name__ == "__main__":
    from backend.pdf_processor import extract_pdf_pages_into_images
    pdf_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.pdf"
    # pdf_file = open(pdf_path, "rb")
    images = extract_pdf_pages_into_images(pdf_path)
    print(f"there are {len(images)} pages in the pdf")

    detector = DocLayoutDetector(model_name="docstructbench", model_path="./models/layout_detection/model_parameters/docstructbench_doclayout_yolo_docstructbench_imgsz1024.pt")

    result = detector.detect(images[0])

    print(result)