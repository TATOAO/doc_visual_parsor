"""
Abstract Base Class for Document Layout Detection

This module defines the abstract interface for all layout detection implementations.
Different detection methods (CV-based, rule-based, etc.) should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from dataclasses import dataclass
from enum import Enum
from models.schemas.docx_schemas import LayoutDetectionResult
from models.schemas.schemas import Section


class BaseLayoutDetector(ABC):
    """
    Abstract base class for document layout detection.
    
    All layout detection implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 **kwargs):
        """
        Initialize the layout detector.
        
        Args:
            confidence_threshold: Minimum confidence for detections
            device: Device to use for computation
            **kwargs: Additional detector-specific parameters
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.is_initialized = False
        
    @abstractmethod
    def _initialize_detector(self) -> None:
        """Initialize the detector (load models, setup resources, etc.)."""
        pass
    
    @abstractmethod
    def _detect_layout(self, 
                      input_data: Any,
                      confidence_threshold: Optional[float] = None,
                      **kwargs) -> LayoutDetectionResult:
        """
        Core detection method to be implemented by subclasses.
        
        Args:
            input_data: Input data (image, document path, etc.)
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutDetectionResult containing detected elements
        """
        pass
    
    def detect(self, 
               input_data: Any,
               confidence_threshold: Optional[float] = None,
               **kwargs) -> LayoutDetectionResult:
        """
        Public interface for layout detection.
        
        Args:
            input_data: Input data (image, document path, etc.)
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            LayoutDetectionResult containing detected elements
        """
        if not self.is_initialized:
            self._initialize_detector()
            self.is_initialized = True
        
        return self._detect_layout(input_data, confidence_threshold, **kwargs)
    
    def detect_batch(self, 
                    input_data_list: List[Any],
                    confidence_threshold: Optional[float] = None,
                    **kwargs) -> List[LayoutDetectionResult]:
        """
        Detect layout in multiple inputs.
        
        Args:
            input_data_list: List of input data
            confidence_threshold: Override default confidence threshold
            **kwargs: Additional detection parameters
            
        Returns:
            List of LayoutDetectionResult objects
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.detect(input_data, confidence_threshold, **kwargs)
                results.append(result)
            except Exception as e:
                # Create empty result for failed detection
                print(f"Failed to process input {input_data}: {str(e)}")
                results.append(LayoutDetectionResult([]))
        
        return results
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.
        
        Returns:
            List of supported file extensions or format descriptions
        """
        pass
    
    @abstractmethod
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.
        
        Returns:
            Dictionary containing detector information
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate if input data is supported by this detector.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is supported, False otherwise
        """
        # Default implementation - subclasses can override
        return True
    
    def visualize(self, 
                  input_data: Any,
                  result: Optional[LayoutDetectionResult] = None,
                  save_path: Optional[str] = None,
                  **kwargs) -> Optional[np.ndarray]:
        """
        Visualize detection results (optional implementation).
        
        Args:
            input_data: Original input data
            result: Detection result (if None, will run detection)
            save_path: Path to save visualization
            **kwargs: Additional visualization parameters
            
        Returns:
            Visualized image as numpy array (if applicable)
        """
        # Default implementation - subclasses can override
        print("Visualization not implemented for this detector type")
        return None 


class BaseSectionDetector(BaseLayoutDetector, ABC):
    """
    Abstract base class for section detection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_section_tree(self, input_data: Any) ->Section:
        """
        Generate a section tree from the input data.
        """
        pass