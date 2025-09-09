"""
Lightweight ONNX-based Document Layout Detection Module

This module provides a lightweight implementation of document layout detection using
ONNX Runtime instead of PyTorch, significantly reducing memory footprint and dependencies.
"""

import logging
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Any
from PIL import Image
import io

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("Please install onnxruntime: pip install onnxruntime")

try:
    import fitz  # PyMuPDF for PDF support
except ImportError:
    fitz = None

from .schemas import (
    LayoutExtractionResult, 
    LayoutElement, 
    BoundingBox,
    ElementType
)
from .utils import sort_elements_by_position, filter_redundant_boxes

logger = logging.getLogger(__name__)

# Mapping from DocLayout-YOLO class IDs to our standardized ElementType
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


class ONNXLayoutDetector:
    """
    Lightweight ONNX-based Document Layout Detector.
    
    This class provides CV-based layout detection using ONNX Runtime instead of PyTorch,
    resulting in much smaller memory footprint and faster startup times.
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = "auto",
                 confidence_threshold: float = 0.25,
                 image_size: int = 1024,
                 pdf_dpi: int = 150,
                 **kwargs):
        """
        Initialize the ONNX-based layout detector.
        
        Args:
            model_path: Path to the ONNX model file
            device: Device to use ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            confidence_threshold: Minimum confidence for detections
            image_size: Input image size for the model
            pdf_dpi: DPI resolution for PDF to image conversion
            **kwargs: Additional parameters
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_path = model_path
        self.image_size = image_size
        self.pdf_dpi = pdf_dpi
        self.session = None
        self.input_name = None
        self.output_names = None
        self.is_initialized = False
        
        # Check PDF support
        if fitz is None:
            logger.warning("PyMuPDF not available. PDF support disabled.")
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
    
    def _initialize_detector(self) -> None:
        """Initialize the detector by loading the ONNX model."""
        try:
            # Determine device
            if self.device == "auto":
                providers = ['CPUExecutionProvider']
                if ort.get_device() == 'GPU':
                    providers.insert(0, 'CUDAExecutionProvider')
            elif self.device.startswith("cuda"):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX model loaded successfully from {self.model_path}")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output names: {self.output_names}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX detector: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert BGR to RGB
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        padded = padded.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        return padded
    
    def _postprocess_outputs(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[LayoutElement]:
        """
        Postprocess ONNX model outputs to extract layout elements.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detected layout elements
        """
        elements = []
        element_id = 0
        
        # Extract detections from outputs
        # Assuming YOLO-style output format: [batch, num_detections, 85]
        # where 85 = 4 (bbox) + 1 (confidence) + 80 (class probabilities)
        if len(outputs) > 0:
            detections = outputs[0]  # Shape: [1, num_detections, 85]
            
            print(f"Debug: Raw output shape: {detections.shape}")
            print(f"Debug: First few detections: {detections[:3] if len(detections) > 0 else 'No detections'}")
            
            if len(detections.shape) == 3:
                detections = detections[0]  # Remove batch dimension
            
            for detection in detections:
                # Extract bounding box (normalized coordinates)
                x_center, y_center, width, height = detection[:4]
                confidence = detection[4]
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    continue
                
                # Get class with highest probability
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                
                # Apply softmax to normalize class scores to probabilities
                class_scores_exp = np.exp(class_scores - np.max(class_scores))  # Subtract max for numerical stability
                class_probabilities = class_scores_exp / np.sum(class_scores_exp)
                class_confidence = class_probabilities[class_id]
                
                # Skip if class confidence is too low
                if class_confidence < self.confidence_threshold:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                orig_h, orig_w = original_shape
                
                # Convert from center format to corner format
                x1 = (x_center - width / 2) * orig_w
                y1 = (y_center - height / 2) * orig_h
                x2 = (x_center + width / 2) * orig_w
                y2 = (y_center + height / 2) * orig_h
                
                # Clamp coordinates to image boundaries to prevent negative values
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # Ensure x2 > x1 and y2 > y1
                if x2 <= x1:
                    x2 = x1 + 1
                if y2 <= y1:
                    y2 = y1 + 1
                
                # Final validation - ensure coordinates are within image bounds
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(x1 + 1, min(x2, orig_w))
                y2 = max(y1 + 1, min(y2, orig_h))
                
                # Debug output
                print(f"Debug coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}, orig_w={orig_w}, orig_h={orig_h}")
                
                # Map class ID to element type
                element_type = DOCLAYOUT_CLASS_MAPPING.get(class_id, ElementType.UNKNOWN)
                
                # Create bounding box
                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                
                # Calculate final confidence and clamp to valid range
                final_confidence = float(confidence * class_confidence)
                final_confidence = max(0.0, min(1.0, final_confidence))  # Clamp to [0, 1]
                
                # Create layout element
                element = LayoutElement(
                    id=element_id,
                    element_type=element_type,
                    confidence=final_confidence,
                    bbox=bbox,
                    metadata={
                        'model_class_id': int(class_id),
                        'detection_method': 'onnx_yolo',
                        'source_type': 'single_image'
                    }
                )
                
                elements.append(element)
                element_id += 1
        
        return elements
    
    def detect_layout(self, 
                      input_data: InputDataType,
                      confidence_threshold: Optional[float] = None,
                      **kwargs) -> LayoutExtractionResult:
        """
        Detect layout elements in input data.
        
        Args:
            input_data: Input data (file path, image array, or PDF)
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected elements
        """
        if not self.is_initialized:
            self._initialize_detector()
        
        # Use provided threshold or default
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # Determine input type and process accordingly
        if isinstance(input_data, (str, Path)):
            input_path = str(input_data)
            if input_path.lower().endswith('.pdf'):
                return self._detect_layout_pdf(input_data, threshold, self.image_size, **kwargs)
            else:
                return self._detect_layout_image(input_data, threshold, self.image_size, **kwargs)
        elif isinstance(input_data, (bytes, io.BytesIO, io.BufferedReader)):
            # Assume PDF for binary data
            return self._detect_layout_pdf(input_data, threshold, self.image_size, **kwargs)
        else:
            # Assume image data
            return self._detect_layout_image(input_data, threshold, self.image_size, **kwargs)
    
    def _detect_layout_pdf(self, 
                          pdf_input: InputDataType,
                          confidence_threshold: float,
                          image_size: int,
                          **kwargs) -> LayoutExtractionResult:
        """
        Detect layout in PDF document.
        
        Args:
            pdf_input: PDF input data
            confidence_threshold: Confidence threshold
            image_size: Image size for detection
            **kwargs: Additional parameters
            
        Returns:
            LayoutExtractionResult containing detected elements
        """
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF processing")
        
        try:
            # Open PDF document
            if isinstance(pdf_input, (str, Path)):
                doc = fitz.open(str(pdf_input))
            else:
                # Handle bytes or file-like objects
                doc = fitz.open(stream=pdf_input, filetype="pdf")
            
            all_elements = []
            element_id = 0
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(self.pdf_dpi / 72, self.pdf_dpi / 72)  # 72 DPI is default
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(img_data))
                image_array = np.array(image)
                
                # Detect layout on this page
                page_result = self._detect_layout_image(image_array, confidence_threshold, image_size, **kwargs)
                
                # Update element IDs and add page number
                for element in page_result.elements:
                    element.id = element_id
                    element.metadata = element.metadata or {}
                    element.metadata['page_number'] = page_num
                    element.metadata['source_type'] = 'pdf_page'
                    all_elements.append(element)
                    element_id += 1
            
            doc.close()
            
            # Filter out redundant boxes
            all_elements = filter_redundant_boxes(all_elements)
            
            # Sort elements in natural reading order
            all_elements = sort_elements_by_position(all_elements)
            
            # Reassign IDs
            for i, element in enumerate(all_elements):
                element.id = i
            
            return LayoutExtractionResult(elements=all_elements)
            
        except Exception as e:
            logger.error(f"PDF ONNX detection failed: {str(e)}")
            raise
    
    def _detect_layout_image(self, 
                           input_data: Any,
                           confidence_threshold: float,
                           image_size: int,
                           **kwargs) -> LayoutExtractionResult:
        """
        Core ONNX-based layout detection method for single images.
        
        Args:
            input_data: Input image (file path, numpy array, or PIL Image)
            confidence_threshold: Confidence threshold
            image_size: Image size for detection
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected elements
        """
        # Load and preprocess image
        if isinstance(input_data, (str, Path)):
            image = cv2.imread(str(input_data))
            if image is None:
                raise ValueError(f"Could not load image from {input_data}")
        elif isinstance(input_data, Image.Image):
            image = np.array(input_data)
        else:
            image = input_data.copy()
        
        original_shape = image.shape[:2]  # (height, width)
        
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        try:
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Postprocess outputs
            elements = self._postprocess_outputs(outputs, original_shape)
            
            # Filter out redundant boxes
            elements = filter_redundant_boxes(elements)
            
            # Sort elements in natural reading order
            elements = sort_elements_by_position(elements)
            
            # Reassign IDs
            for i, element in enumerate(elements):
                element.id = i
            
            return LayoutExtractionResult(elements=elements)
            
        except Exception as e:
            logger.error(f"ONNX detection failed: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported input formats."""
        formats = ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
        if fitz is not None:
            formats.append("pdf")
        return formats
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'session') and self.session is not None:
            del self.session


# python -m doc_chunking.src.detection
if __name__ == "__main__":
    detector = ONNXLayoutDetector(model_path="model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx")
    detector._initialize_detector()
    result = detector.detect_layout("/Users/tatoao_mini/Work/Kindee/合同/合同脱敏AI测试/3800.pdf")
    print(result)