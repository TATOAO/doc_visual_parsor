#!/usr/bin/env python3
"""
Example script for using exported ONNX model for DocLayout-YOLO inference.

This script demonstrates how to:
1. Load an ONNX model
2. Preprocess input images
3. Run inference
4. Post-process results
5. Visualize and save results

Usage:
    python onnx_inference_example.py --model model.onnx --image-path image.png --output outputs/
"""

import os
import cv2
import numpy as np
import argparse
import time
from typing import List, Tuple, Optional
import onnxruntime as ort

from .schemas import ElementType, LayoutExtractionResult, LayoutElement, BoundingBox

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


class ONNXDocLayoutYOLO:
    """ONNX inference wrapper for DocLayout-YOLO model."""
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None, device: str = "auto"):
        """
        Initialize ONNX model.
        
        Args:
            model_path (str): Path to the ONNX model file
            providers (List[str]): ONNX Runtime execution providers
        """
        self.model_path = model_path
        self.input_size = 1024  # Default input size
        self.device = device


        # Set up ONNX Runtime providers
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
        
        if device == "cuda":
            providers.append('CUDAExecutionProvider')
        elif device == "cpu":
            providers.append('CPUExecutionProvider')
        elif device == "auto":
            providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        else:
            raise ValueError(f"Invalid device: {device}")

        # remove the duplicate providers
        self.providers = list(set(providers))
        
        # Filter available providers
        available_providers = ort.get_available_providers()
        self.providers = [p for p in providers if p in available_providers]
        
        print(f"Available providers: {available_providers}")
        print(f"Using providers: {self.providers}")
        
        # Load the ONNX model
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:  # NCHW format
            self.input_size = input_shape[2]  # Height
            self.input_channels = input_shape[1]
        
        print(f"Model input name: {self.input_name}")
        print(f"Model output names: {self.output_names}")
        print(f"Input size: {self.input_size}x{self.input_size}")
        print(f"Input channels: {self.input_channels}")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Preprocess image for model inference.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Preprocessed image, original image, scale factor
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        h, w = image.shape[:2]
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to RGB and normalize
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        input_tensor = np.transpose(padded, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor, original_image, scale
    
    def postprocess_output(self, outputs: List[np.ndarray], original_image: np.ndarray, 
                          scale: float, conf_threshold: float = 0.2) -> List[dict]:
        """
        Post-process model outputs to get bounding boxes and labels.
        
        Args:
            outputs (List[np.ndarray]): Model outputs
            original_image (np.ndarray): Original input image
            scale (float): Scale factor used in preprocessing
            conf_threshold (float): Confidence threshold for filtering detections
            
        Returns:
            List[dict]: List of detection results
        """
        # Assuming the model outputs predictions in the format [batch, num_detections, 6]
        # where 6 = [x1, y1, x2, y2, conf, class]
        predictions = outputs[0]  # Take first output
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        detections = []
        h, w = original_image.shape[:2]
        
        for pred in predictions:
            if len(pred) >= 6:
                x1, y1, x2, y2, conf, cls = pred[:6]
                
                # Filter by confidence
                if conf < conf_threshold:
                    continue
                
                # Scale coordinates back to original image size
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                
                # Clip coordinates to image bounds
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.get_class_name(int(cls))
                })
        
        return detections
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get class name from class ID using the standardized mapping.
        
        Args:
            class_id (int): Class ID
            
        Returns:
            str: Class name
        """
        if class_id in DOCLAYOUT_CLASS_MAPPING:
            return DOCLAYOUT_CLASS_MAPPING[class_id].value
        else:
            return f'class_{class_id}'
    
    def predict(self, image_path: str, conf_threshold: float = 0.2) -> List[dict]:
        """
        Run inference on an image.
        
        Args:
            image_path (str): Path to the input image
            conf_threshold (float): Confidence threshold
            
        Returns:
            List[dict]: Detection results
        """
        # Preprocess image
        input_tensor, original_image, scale = self.preprocess_image(image_path)
        
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.3f}s")
        
        # Post-process results
        detections = self.postprocess_output(outputs, original_image, scale, conf_threshold)
        
        return detections
    
    def detect_layout(self, input_data: str, confidence_threshold: float = 0.2, **kwargs) -> LayoutExtractionResult:
        """
        Detect layout elements in an image and return standardized results.
        
        Args:
            input_data (str): Path to the input image
            confidence_threshold (float): Confidence threshold for detections
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            LayoutExtractionResult: Standardized layout detection results
        """
        # Get raw predictions
        detections = self.predict(input_data, confidence_threshold)
        
        # Convert to LayoutElement objects
        elements = []
        for i, detection in enumerate(detections):
            bbox = BoundingBox(
                x1=float(detection['bbox'][0]),
                y1=float(detection['bbox'][1]),
                x2=float(detection['bbox'][2]),
                y2=float(detection['bbox'][3])
            )
            
            element = LayoutElement(
                id=i,
                element_type=DOCLAYOUT_CLASS_MAPPING.get(detection['class_id'], ElementType.UNKNOWN),
                bbox=bbox,
                confidence=detection['confidence']
            )
            elements.append(element)
        
        return LayoutExtractionResult(
            elements=elements,
            metadata={
                'model': 'ONNXDocLayoutYOLO',
                'confidence_threshold': confidence_threshold,
                'total_detections': len(detections)
            }
        )
    
    def visualize_results(self, image_path: str, detections: List[dict], 
                         output_path: str, line_width: int = 2, font_size: float = 0.5):
        """
        Visualize detection results on the image.
        
        Args:
            image_path (str): Path to the original image
            detections (List[dict]): Detection results
            output_path (str): Path to save the visualized image
            line_width (int): Line width for bounding boxes
            font_size (float): Font size for labels
        """
        # Load original image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), line_width)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
            
            # Draw label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 2)
        
        # Save result
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ONNX DocLayout-YOLO Inference Example')
    parser.add_argument('--model', required=True, type=str, 
                       help='Path to the ONNX model file')
    parser.add_argument('--image-path', required=True, type=str,
                       help='Path to the input image')
    parser.add_argument('--output', default='outputs', type=str,
                       help='Output directory for results')
    parser.add_argument('--conf', default=0.2, type=float,
                       help='Confidence threshold (default: 0.2)')
    parser.add_argument('--line-width', default=2, type=int,
                       help='Line width for bounding boxes (default: 2)')
    parser.add_argument('--font-size', default=0.5, type=float,
                       help='Font size for labels (default: 0.5)')
    parser.add_argument('--providers', nargs='+', default=None,
                       help='ONNX Runtime execution providers')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    if not os.path.exists(args.image_path):
        print(f"❌ Image file not found: {args.image_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Initialize ONNX model
        print("Loading ONNX model...")
        model = ONNXDocLayoutYOLO(args.model, args.providers)
        
        # Run inference
        print(f"Running inference on: {args.image_path}")
        detections = model.predict(args.image_path, args.conf)
        
        # Print results
        print(f"\nFound {len(detections)} detections:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f} "
                  f"at [{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]")
        
        # Visualize results
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_path = os.path.join(args.output, f"{image_name}_onnx_result.jpg")
        
        model.visualize_results(args.image_path, detections, output_path, 
                               args.line_width, args.font_size)
        
        print(f"\n✅ Inference completed successfully!")
        print(f"📁 Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


# python onnx_inference_onnx_layout_detector.py --model docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx --image-path test.png --output outputs

# python -m doc_chunking.src.onnx_layout_detector --model model_parameters/layout_detection/docstructbench_doclayout_yolo_docstructbench_imgsz1024.onnx --image-path image.png --output outputs
if __name__ == "__main__":
    exit(main())
