"""
ONNX-based table detection module.

This module provides table detection using ONNX models without PyTorch dependencies.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from .schemas import TableElement, TableType
from ..schemas import BoundingBox, ElementType

logger = logging.getLogger(__name__)


class ONNXTableDetector:
    """
    ONNX-based table detector for identifying table regions in documents.
    
    This class uses ONNX models to detect table regions without requiring PyTorch.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.3,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize the ONNX table detector.
        
        Args:
            model_path: Path to ONNX model file (optional, will use default if not provided)
            confidence_threshold: Minimum confidence for detections
            device: Device to use ('auto', 'cpu', 'cuda')
            **kwargs: Additional parameters
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_size = 640  # Default input size
        
        # Initialize model if path provided or use default
        if self.model_path:
            if os.path.exists(self.model_path):
                self._initialize_model()
        else:
            # Use default model - will be downloaded in _initialize_model
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the ONNX model."""
        try:
            # Setup providers
            providers = self._get_onnx_providers(self.device)

            if not self.model_path or not os.path.exists(self.model_path):
                from doc_chunking.utils.download_onnx import download_onnx
                self.model_path = download_onnx(model_name='paddle_table_cell_det.onnx')

            
            # Load ONNX model
            logger.info(f"Loading table detection model from: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            # Get input/output information
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Log input/output info for debugging
            logger.info(f"Model inputs: {self.input_names}")
            logger.info(f"Model outputs: {self.output_names}")
            for i, input_info in enumerate(self.session.get_inputs()):
                logger.info(f"Input {i}: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            
            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) == 4:  # NCHW format
                self.input_size = input_shape[2]  # Height
            
            logger.info(f"Table detection model loaded successfully")
            logger.info(f"Input size: {self.input_size}x{self.input_size}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize table detection model: {str(e)}")
            raise
    
    def _get_onnx_providers(self, device: str) -> List[str]:
        """Get ONNX Runtime providers based on device."""
        available_providers = ort.get_available_providers()
        
        # Default provider priority: CUDA > CoreML > CPU
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
        
        # Configure providers based on device preference
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif device == "cpu":
            providers = ['CPUExecutionProvider']
        elif device == "auto":
            # Auto-detect: prefer CUDA if available, otherwise CPU
            available_providers = ort.get_available_providers()
            if 'CoreMLExecutionProvider' in available_providers:
                providers = ['CoreMLExecutionProvider']
            elif 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        return providers

        
            
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for table detection.
        
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
    
    def _postprocess_detections(self, 
                               outputs: List[np.ndarray], 
                               original_shape: Tuple[int, int],
                               scale_x: float, 
                               scale_y: float) -> List[TableElement]:
        """
        Postprocess model outputs to extract table detections.
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
            
        Returns:
            List of detected table elements
        """
        tables = []
        
        if len(outputs) >= 2:
            # PaddlePaddle detection output format:
            # outputs[0]: [num_detections, 6] - [x1, y1, x2, y2, confidence, class_id]
            # outputs[1]: [1] - number of valid detections
            
            detections = outputs[0]  # [num_detections, 6]
            num_detections = int(outputs[1][0]) if len(outputs[1]) > 0 else len(detections)
            
            for i in range(min(num_detections, len(detections))):
                detection = detections[i]
                if len(detection) >= 6:
                    # Format: [class_id, confidence, x1, y1, x2, y2]
                    class_id, confidence, x1, y1, x2, y2 = detection[:6]
                    
                    if confidence >= self.confidence_threshold:
                        # The coordinates are already in original image space for this model
                        # No scaling needed - just clamp to image bounds
                        x1 = max(0, min(x1, original_shape[1]))
                        y1 = max(0, min(y1, original_shape[0]))
                        x2 = max(0, min(x2, original_shape[1]))
                        y2 = max(0, min(y2, original_shape[0]))
                        
                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Create table element
                        table = TableElement(
                            id=len(tables),
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(confidence),
                            metadata={
                                'detection_method': 'onnx_table_detection',
                                'class_id': int(class_id),
                                'original_shape': original_shape
                            }
                        )
                        
                        tables.append(table)
        
        return tables
    
    def detect_tables(self, 
                     image: Union[str, np.ndarray, Path],
                     confidence_threshold: Optional[float] = None) -> List[TableElement]:
        """
        Detect tables in an image.
        
        Args:
            image: Input image (file path, numpy array, or Path)
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of detected table elements
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        # Use provided threshold or default
        conf_thresh = confidence_threshold or self.confidence_threshold
        
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            img = image.copy()
        
        original_shape = img.shape[:2]  # (height, width)
        
        # Preprocess image
        input_tensor, scale_x, scale_y = self._preprocess_image(img)
        
        try:
            # Prepare input feed dict
            input_feed = {}
            
            # Add image input
            if 'image' in self.input_names:
                input_feed['image'] = input_tensor
            else:
                # Fallback to first input
                input_feed[self.input_names[0]] = input_tensor
            
            # Add scale factor if required
            if 'scale_factor' in self.input_names:
                # Create scale factor tensor [scale_y, scale_x]
                scale_factor = np.array([[scale_y, scale_x]], dtype=np.float32)
                input_feed['scale_factor'] = scale_factor
            
            # Add im_shape if required (original image shape)
            if 'im_shape' in self.input_names:
                # Create im_shape tensor [height, width]
                im_shape = np.array([[original_shape[0], original_shape[1]]], dtype=np.float32)
                input_feed['im_shape'] = im_shape
            
            # Run inference
            outputs = self.session.run(self.output_names, input_feed)
            
            # Postprocess outputs
            tables = self._postprocess_detections(outputs, original_shape, scale_x, scale_y)
            
            # Filter overlapping detections (NMS)
            tables = self._non_max_suppression(tables)
            
            # Reassign IDs
            for i, table in enumerate(tables):
                table.id = i
            
            logger.info(f"Detected {len(tables)} tables")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {str(e)}")
            raise
    
    def _non_max_suppression(self, tables: List[TableElement], iou_threshold: float = 0.5) -> List[TableElement]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            tables: List of table detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of table detections
        """
        if len(tables) <= 1:
            return tables
        
        # Sort by confidence (descending)
        tables = sorted(tables, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while tables:
            # Take the highest confidence detection
            current = tables.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for table in tables:
                if self._calculate_iou(current.bbox, table.bbox) < iou_threshold:
                    remaining.append(table)
            tables = remaining
        
        return keep
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def crop_table_region(self, image: np.ndarray, table: TableElement) -> np.ndarray:
        """
        Crop table region from image.
        
        Args:
            image: Source image
            table: Table element with bounding box
            
        Returns:
            Cropped table region
        """
        x1, y1, x2, y2 = table.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        return image[y1:y2, x1:x2]
    
    def visualize_detections(self, 
                            image: Union[str, np.ndarray, Path], 
                            detections: List[TableElement],
                            output_path: Optional[str] = None,
                            show_confidence: bool = True,
                            box_color: str = 'red',
                            text_color: str = 'white',
                            box_thickness: int = 2,
                            font_size: int = 8) -> None:
        """
        Visualize detected cells on the image.
        
        Args:
            image: Input image (file path, numpy array, or Path)
            detections: List of detected cell elements
            output_path: Path to save the visualization (optional)
            show_confidence: Whether to show confidence scores
            box_color: Color of bounding boxes
            text_color: Color of confidence text
            box_thickness: Thickness of bounding box lines
            font_size: Font size for confidence text
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(img_rgb)
        ax.set_title(f'Cell Detection Results - {len(detections)} cells detected', fontsize=16)
        
        # Draw bounding boxes
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=box_thickness,
                edgecolor=box_color,
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add confidence text if requested
            if show_confidence:
                confidence_text = f'{detection.confidence:.3f}'
                ax.text(
                    x1, y1 - 5,
                    confidence_text,
                    color=text_color,
                    fontsize=font_size,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.7),
                    verticalalignment='top'
                )
            
            # Add cell ID
            ax.text(
                x1 + 5, y1 + 15,
                f'Cell {i+1}',
                color=text_color,
                fontsize=font_size,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7)
            )
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add statistics text
        stats_text = f'Total cells: {len(detections)}\n'
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            max_conf = max(d.confidence for d in detections)
            min_conf = min(d.confidence for d in detections)
            stats_text += f'Avg confidence: {avg_conf:.3f}\n'
            stats_text += f'Max confidence: {max_conf:.3f}\n'
            stats_text += f'Min confidence: {min_conf:.3f}'
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_detector_info(self) -> dict:
        """Get information about the cell detector."""
        return {
            'detector_type': 'onnx_cell_detection',
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.input_size,
            'model_loaded': self.session is not None,
            'providers': self.session.get_providers() if self.session else []
        }

# python -m doc_chunking.table_parsing.table_detector
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    detector = ONNXTableDetector(model_path=None, device="auto")
    cells = detector.detect_tables("demo_table_wo_line.png")
    
    print(f"\n=== Cell Detection Results ===")
    print(f"Found {len(cells)} cells")
    for i, cell in enumerate(cells):
        print(f"Cell {i+1}: bbox={cell.bbox}, confidence={cell.confidence:.3f}")
    print("=== End Results ===\n")
    
    # Create visualization
    print("Creating visualization...")
    detector.visualize_detections(
        "demo_table_wo_line.png", 
        cells, 
        output_path="cell_detection_visualization.png",
        show_confidence=True,
        box_color='red',
        box_thickness=2
    )