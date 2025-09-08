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

from ..base.base_layout_extractor import BaseLayoutExtractor
from doc_chunking.schemas.layout_schemas import (
    LayoutExtractionResult, 
    LayoutElement, 
    BoundingBox,
    ElementType
)
from ..utils.layout_processing import sort_elements_by_position, filter_redundant_boxes

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

class ONNXLayoutDetector(BaseLayoutExtractor):
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
        super().__init__(confidence_threshold=confidence_threshold, device=device, **kwargs)
        
        self.model_path = model_path
        self.image_size = image_size
        self.pdf_dpi = pdf_dpi
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # Check PDF support
        if fitz is None:
            logger.warning("PyMuPDF not available. PDF support disabled.")
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
    
    def _initialize_detector(self) -> None:
        """Initialize the detector by loading the ONNX model."""
        try:
            # Setup device
            self.device = self._setup_device(self.device)
            
            # Configure ONNX Runtime providers
            providers = self._get_onnx_providers()
            
            # Load ONNX model
            logger.info(f"Loading ONNX model from: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=providers
            )
            
            # Get input/output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output names: {self.output_names}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX detector: {str(e)}")
            raise
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for ONNX Runtime."""
        if device == "auto":
            # Check available providers
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                device = "cuda"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return device
    
    def _get_onnx_providers(self) -> List[str]:
        """Get ONNX Runtime providers based on device."""
        available_providers = ort.get_available_providers()
        
        if self.device == "cuda" and 'CUDAExecutionProvider' in available_providers:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX model inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to CHW format
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def _postprocess_outputs(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[LayoutElement]:
        """
        Postprocess ONNX model outputs to extract bounding boxes and classes.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detected layout elements
        """
        elements = []
        
        # YOLO ONNX outputs typically contain:
        # - boxes: [batch, num_detections, 4] (x1, y1, x2, y2)
        # - scores: [batch, num_detections]
        # - classes: [batch, num_detections]
        
        if len(outputs) >= 3:
            boxes = outputs[0]  # [batch, num_detections, 4]
            scores = outputs[1]  # [batch, num_detections]
            classes = outputs[2]  # [batch, num_detections]
            
            # Remove batch dimension
            boxes = boxes[0]  # [num_detections, 4]
            scores = scores[0]  # [num_detections]
            classes = classes[0]  # [num_detections]
            
            # Scale boxes back to original image size
            scale_x = original_shape[1] / self.image_size
            scale_y = original_shape[0] / self.image_size
            
            for i in range(len(boxes)):
                if scores[i] >= self.confidence_threshold:
                    # Scale coordinates
                    x1, y1, x2, y2 = boxes[i]
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                    
                    # Map class ID to ElementType
                    cls_id = int(classes[i])
                    element_type = DOCLAYOUT_CLASS_MAPPING.get(cls_id, ElementType.UNKNOWN)
                    
                    # Create bounding box
                    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    
                    # Create layout element
                    element = LayoutElement(
                        id=len(elements),
                        element_type=element_type,
                        confidence=float(scores[i]),
                        bbox=bbox,
                        metadata={
                            'model_class_id': cls_id,
                            'detection_method': 'onnx_yolo',
                            'source_type': 'single_image'
                        }
                    )
                    
                    elements.append(element)
        
        return elements
    
    def _detect_layout(self, 
                      input_data: Any,
                      confidence_threshold: Optional[float] = None,
                      image_size: Optional[int] = None,
                      **kwargs) -> LayoutExtractionResult:
        """
        Core ONNX-based layout detection method.
        
        Args:
            input_data: Input data (image file path, numpy array, PIL Image, or PDF file path)
            confidence_threshold: Override default confidence threshold
            image_size: Override default image size
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutExtractionResult containing detected elements
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call _initialize_detector() first.")
        
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
        Convert a PDF page to image for ONNX analysis.
        
        Args:
            doc: PyMuPDF Document object
            page_num: Page number (0-indexed)
            
        Returns:
            Tuple of (page image as numpy array, scale factor) or (None, 1.0) if failed
        """
        try:
            page = doc[page_num]
            
            # Calculate zoom factor based on desired DPI
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
            # Load PDF document
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
                'detection_method': 'onnx_yolo_pdf',
                'total_elements': len(all_elements),
                'document_type': 'pdf',
                'page_count': doc.page_count,
                'pdf_dpi': self.pdf_dpi
            }
            
            doc.close()
            
            return LayoutExtractionResult(elements=all_elements, metadata=metadata)
            
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
        formats = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            'PIL.Image', 'numpy.ndarray', 'file_path'
        ]
        
        if fitz is not None:
            formats.append('.pdf')
        
        return formats
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the ONNX detector."""
        return {
            'detector_type': 'onnx_based',
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'image_size': self.image_size,
            'pdf_dpi': self.pdf_dpi,
            'pdf_support': fitz is not None,
            'supported_formats': self.get_supported_formats(),
            'class_mapping': {k: v.value for k, v in DOCLAYOUT_CLASS_MAPPING.items()},
            'onnx_providers': self.session.get_providers() if self.session else []
        }
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate if input data is supported by ONNX detector."""
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


# Example usage
if __name__ == "__main__":
    # Example usage - you'll need to provide the path to your ONNX model
    model_path = "path/to/your/model.onnx"  # Replace with actual path
    
    if os.path.exists(model_path):
        detector = ONNXLayoutDetector(model_path=model_path)
        detector._initialize_detector()
        
        # Test with a sample image
        test_image = "tests/test_data/1-1 买卖合同（通用版）.pdf"
        if os.path.exists(test_image):
            result = detector._detect_layout(test_image)
            print(f"Detected {len(result.elements)} elements")
            
            # Save results
            import json
            with open("test_onnx_result.json", "w") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    else:
        print(f"ONNX model not found at: {model_path}")
        print("Please convert your PyTorch model to ONNX format first using convert_to_onnx.py")
