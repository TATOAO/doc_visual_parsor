"""
Layout Detector Factory

This module provides a factory class for creating different types of layout detectors
based on input format or user preference.
"""

import logging
from pathlib import Path
from typing import Union, Optional, Any, Dict, List
from enum import Enum

from .base_detector import BaseLayoutDetector
from .cv_detector import CVLayoutDetector, DocLayoutDetector  # Backward compatibility
from .document_detector import DocumentLayoutDetector

logger = logging.getLogger(__name__)

class DetectorType(Enum):
    """Available detector types."""
    CV_BASED = "cv_based"
    DOCUMENT_NATIVE = "document_native"
    AUTO = "auto"

class LayoutDetectorFactory:
    """
    Factory class for creating layout detectors.
    
    This factory can automatically select the appropriate detector type
    based on input format or create specific detector types as requested.
    """
    
    @staticmethod
    def create_detector(
        detector_type: Union[DetectorType, str] = DetectorType.AUTO,
        **kwargs
    ) -> BaseLayoutDetector:
        """
        Create a layout detector.
        
        Args:
            detector_type: Type of detector to create
            **kwargs: Arguments to pass to the detector constructor
            
        Returns:
            BaseLayoutDetector instance
        """
        if isinstance(detector_type, str):
            detector_type = DetectorType(detector_type.lower())
        
        if detector_type == DetectorType.CV_BASED:
            return CVLayoutDetector(**kwargs)
        elif detector_type == DetectorType.DOCUMENT_NATIVE:
            return DocumentLayoutDetector(**kwargs)
        elif detector_type == DetectorType.AUTO:
            # Return CV-based as default for auto mode
            # Will be refined by auto_detect_best_detector
            return CVLayoutDetector(**kwargs)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    @staticmethod
    def auto_detect_best_detector(
        input_data: Any,
        **kwargs
    ) -> BaseLayoutDetector:
        """
        Automatically select the best detector for the given input.
        
        Args:
            input_data: Input data to analyze
            **kwargs: Arguments to pass to the detector constructor
            
        Returns:
            Best suited BaseLayoutDetector instance
        """
        # Determine input type
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if path.suffix.lower() == '.docx':
                logger.info("Auto-selected document-native detector for .docx file")
                return DocumentLayoutDetector(**kwargs)
            elif path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']:
                logger.info("Auto-selected CV-based detector for image/PDF file")
                return CVLayoutDetector(**kwargs)
        elif hasattr(input_data, 'getvalue'):
            # File-like object - check content type if available
            if hasattr(input_data, 'content_type'):
                if 'officedocument.wordprocessingml' in input_data.content_type:
                    logger.info("Auto-selected document-native detector based on content type")
                    return DocumentLayoutDetector(**kwargs)
            # Default to trying both
            logger.info("Auto-selected CV-based detector for file-like object")
            return CVLayoutDetector(**kwargs)
        elif isinstance(input_data, bytes):
            # Check file signature
            if input_data.startswith(b'PK'):  # ZIP-based format (likely DOCX)
                logger.info("Auto-selected document-native detector for ZIP-based file")
                return DocumentLayoutDetector(**kwargs)
        
        # Default to CV-based detector
        logger.info("Auto-selected CV-based detector as default")
        return CVLayoutDetector(**kwargs)
    
    @staticmethod
    def get_available_detectors() -> Dict[str, Dict[str, Any]]:
        """
        Get information about available detectors.
        
        Returns:
            Dictionary with detector information
        """
        detectors = {}
        
        # CV-based detector info
        cv_detector = CVLayoutDetector()
        detectors['cv_based'] = cv_detector.get_detector_info()
        
        # Document-native detector info
        doc_detector = DocumentLayoutDetector()
        detectors['document_native'] = doc_detector.get_detector_info()
        
        return detectors
    
    @staticmethod
    def create_multi_detector(
        detector_types: List[Union[DetectorType, str]],
        **kwargs
    ) -> List[BaseLayoutDetector]:
        """
        Create multiple detectors at once.
        
        Args:
            detector_types: List of detector types to create
            **kwargs: Arguments to pass to detector constructors
            
        Returns:
            List of BaseLayoutDetector instances
        """
        detectors = []
        for det_type in detector_types:
            try:
                detector = LayoutDetectorFactory.create_detector(det_type, **kwargs)
                detectors.append(detector)
            except Exception as e:
                logger.error(f"Failed to create detector {det_type}: {str(e)}")
        
        return detectors

class HybridLayoutDetector(BaseLayoutDetector):
    """
    Hybrid detector that can use multiple detection methods and combine results.
    
    This detector can try multiple approaches and either return the best result
    or combine results from different detectors.
    """
    
    def __init__(self,
                 detector_types: List[Union[DetectorType, str]] = None,
                 combination_strategy: str = "best_confidence",
                 confidence_threshold: float = 0.5,
                 **kwargs):
        """
        Initialize hybrid detector.
        
        Args:
            detector_types: List of detector types to use
            combination_strategy: How to combine results ("best_confidence", "merge", "vote")
            confidence_threshold: Minimum confidence threshold
            **kwargs: Additional arguments
        """
        super().__init__(confidence_threshold=confidence_threshold, **kwargs)
        
        if detector_types is None:
            detector_types = [DetectorType.CV_BASED, DetectorType.DOCUMENT_NATIVE]
        
        self.detector_types = detector_types
        self.combination_strategy = combination_strategy
        self.detectors = []
    
    def _initialize_detector(self) -> None:
        """Initialize all sub-detectors."""
        self.detectors = LayoutDetectorFactory.create_multi_detector(
            self.detector_types,
            confidence_threshold=self.confidence_threshold
        )
        logger.info(f"Hybrid detector initialized with {len(self.detectors)} sub-detectors")
    
    def _detect_layout(self, input_data: Any, confidence_threshold: Optional[float] = None, **kwargs):
        """
        Detect layout using multiple methods and combine results.
        
        Args:
            input_data: Input data
            confidence_threshold: Override confidence threshold
            **kwargs: Additional arguments
            
        Returns:
            Combined LayoutDetectionResult
        """
        results = []
        
        # Run detection with each available detector
        for detector in self.detectors:
            try:
                if detector.validate_input(input_data):
                    result = detector.detect(input_data, confidence_threshold, **kwargs)
                    results.append((detector, result))
                    logger.info(f"Detector {type(detector).__name__} found {len(result.elements)} elements")
                else:
                    logger.info(f"Detector {type(detector).__name__} doesn't support this input")
            except Exception as e:
                logger.warning(f"Detector {type(detector).__name__} failed: {str(e)}")
        
        if not results:
            from .base_detector import LayoutDetectionResult
            return LayoutDetectionResult([])
        
        # Combine results based on strategy
        if self.combination_strategy == "best_confidence":
            return self._select_best_result(results)
        elif self.combination_strategy == "merge":
            return self._merge_results(results)
        elif self.combination_strategy == "vote":
            return self._vote_results(results)
        else:
            # Default to best confidence
            return self._select_best_result(results)
    
    def _select_best_result(self, results: List) -> 'LayoutDetectionResult':
        """Select result with highest average confidence."""
        best_result = None
        best_score = -1
        
        for detector, result in results:
            if result.elements:
                avg_confidence = sum(elem.confidence for elem in result.elements) / len(result.elements)
                if avg_confidence > best_score:
                    best_score = avg_confidence
                    best_result = result
        
        return best_result if best_result else results[0][1]
    
    def _merge_results(self, results: List) -> 'LayoutDetectionResult':
        """Merge all results together."""
        from .base_detector import LayoutDetectionResult
        
        all_elements = []
        element_id = 0
        
        for detector, result in results:
            for element in result.elements:
                # Update element ID and add detector info
                element.id = element_id
                if element.metadata is None:
                    element.metadata = {}
                element.metadata['source_detector'] = type(detector).__name__
                all_elements.append(element)
                element_id += 1
        
        return LayoutDetectionResult(all_elements)
    
    def _vote_results(self, results: List) -> 'LayoutDetectionResult':
        """Use voting to combine results (placeholder implementation)."""
        # For now, just return the merge result
        # In a full implementation, this would identify overlapping detections
        # and use voting to determine the best classification
        return self._merge_results(results)
    
    def get_supported_formats(self) -> List[str]:
        """Get union of all supported formats."""
        all_formats = set()
        for detector in self.detectors:
            all_formats.update(detector.get_supported_formats())
        return list(all_formats)
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the hybrid detector."""
        sub_detector_info = []
        for detector in self.detectors:
            sub_detector_info.append(detector.get_detector_info())
        
        return {
            'detector_type': 'hybrid',
            'combination_strategy': self.combination_strategy,
            'confidence_threshold': self.confidence_threshold,
            'sub_detectors': sub_detector_info,
            'supported_formats': self.get_supported_formats()
        } 