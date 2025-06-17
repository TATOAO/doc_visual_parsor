import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
from enum import Enum
import hashlib
import math

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from models.schemas.layout_schemas import LayoutElement, TextAlignment, ElementType
from .helper import title_number_extraction, TitleNumber


@dataclass
class FeatureWeights:
    """Configurable weights for different features in the element representation"""
    font_size: float = 1.0
    font_style: float = 0.8
    alignment: float = 0.6
    position: float = 0.7
    hierarchy: float = 1.2
    content_semantic: float = 0.9
    element_type: float = 1.0
    spacing: float = 0.5


class ElementCode(BaseModel):
    """Compact representation of a layout element"""
    # Core identification
    element_id: int
    
    # Normalized features (0-1 scale)
    font_size_norm: float = Field(default=0.0, description="Normalized font size")
    font_style_code: int = Field(default=0, description="Encoded font style flags")
    alignment_code: int = Field(default=0, description="Encoded alignment")
    position_code: Tuple[float, float] = Field(default=(0.0, 0.0), description="Normalized position")
    
    # Hierarchical features
    hierarchy_level: int = Field(default=0, description="Detected hierarchy level")
    hierarchy_path: str = Field(default="", description="Hierarchy path string")
    title_number_type: Optional[str] = Field(default=None, description="Type of title numbering")
    
    # Content features
    content_hash: str = Field(default="", description="Hash of content for quick comparison")
    content_embedding: Optional[List[float]] = Field(default=None, description="Semantic embedding")
    content_length_norm: float = Field(default=0.0, description="Normalized content length")
    
    # Element type
    element_type_code: int = Field(default=0, description="Encoded element type")
    
    # Spacing and layout context
    spacing_before: float = Field(default=0.0, description="Normalized space before")
    spacing_after: float = Field(default=0.0, description="Normalized space after")
    indentation: float = Field(default=0.0, description="Normalized indentation")
    
    # Composite features
    structural_signature: str = Field(default="", description="Structural signature for quick matching")
    
    class Config:
        arbitrary_types_allowed = True


class ElementEncoder:
    """
    Encodes layout elements into compact representations for distance-based analysis
    """
    
    def __init__(self, 
                 weights: Optional[FeatureWeights] = None,
                 use_semantic_embeddings: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.weights = weights or FeatureWeights()
        self.use_semantic_embeddings = use_semantic_embeddings and EMBEDDINGS_AVAILABLE
        
        # Initialize semantic model if needed
        self.embedding_model = None
        if self.use_semantic_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.use_semantic_embeddings = False
        
        # Statistics for normalization
        self.font_size_stats = {"min": 8.0, "max": 72.0, "mean": 12.0, "std": 4.0}
        self.position_stats = {"width": 595.0, "height": 842.0}  # A4 default
        self.content_length_stats = {"min": 0, "max": 1000, "mean": 50, "std": 100}
    
    def update_statistics(self, elements: List[LayoutElement]):
        """Update normalization statistics based on the document"""
        font_sizes = []
        positions = []
        content_lengths = []
        widths, heights = [], []
        
        for element in elements:
            # Font sizes
            if element.style and element.style.primary_font and element.style.primary_font.size:
                font_sizes.append(element.style.primary_font.size)
            
            # Positions and dimensions
            if element.bbox:
                positions.append((element.bbox.x1, element.bbox.y1))
                widths.append(element.bbox.width)
                heights.append(element.bbox.height)
            
            # Content lengths
            if element.text:
                content_lengths.append(len(element.text))
        
        # Update font size stats
        if font_sizes:
            self.font_size_stats = {
                "min": min(font_sizes),
                "max": max(font_sizes),
                "mean": np.mean(font_sizes),
                "std": np.std(font_sizes) or 1.0
            }
        
        # Update position stats
        if widths and heights:
            self.position_stats = {
                "width": max(widths) if widths else 595.0,
                "height": max(heights) if heights else 842.0
            }
        
        # Update content length stats
        if content_lengths:
            self.content_length_stats = {
                "min": min(content_lengths),
                "max": max(content_lengths),
                "mean": np.mean(content_lengths),
                "std": np.std(content_lengths) or 1.0
            }
    
    def encode_element(self, element: LayoutElement, context_elements: Optional[List[LayoutElement]] = None) -> ElementCode:
        """Encode a single layout element into compact representation"""
        
        # Initialize code
        code = ElementCode(element_id=element.id)
        
        # Font size normalization
        code.font_size_norm = self._normalize_font_size(element)
        
        # Font style encoding
        code.font_style_code = self._encode_font_style(element)
        
        # Alignment encoding
        code.alignment_code = self._encode_alignment(element)
        
        # Position encoding
        code.position_code = self._encode_position(element)
        
        # Hierarchy analysis
        hierarchy_info = self._analyze_hierarchy(element)
        code.hierarchy_level = hierarchy_info["level"]
        code.hierarchy_path = hierarchy_info["path"]
        code.title_number_type = hierarchy_info["number_type"]
        
        # Content analysis
        content_info = self._analyze_content(element)
        code.content_hash = content_info["hash"]
        code.content_embedding = content_info["embedding"]
        code.content_length_norm = content_info["length_norm"]
        
        # Element type encoding
        code.element_type_code = self._encode_element_type(element)
        
        # Spacing analysis (requires context)
        spacing_info = self._analyze_spacing(element, context_elements)
        code.spacing_before = spacing_info["before"]
        code.spacing_after = spacing_info["after"]
        code.indentation = spacing_info["indentation"]
        
        # Generate structural signature
        code.structural_signature = self._generate_structural_signature(code)
        
        return code
    
    def encode_elements(self, elements: List[LayoutElement]) -> List[ElementCode]:
        """Encode multiple elements with context"""
        # Update statistics first
        self.update_statistics(elements)
        
        # Encode each element with context
        codes = []
        for i, element in enumerate(elements):
            context = elements  # Full context
            code = self.encode_element(element, context)
            codes.append(code)
        
        return codes
    
    def calculate_distance(self, code1: ElementCode, code2: ElementCode, distance_type: str = "weighted") -> float:
        """Calculate distance between two element codes"""
        
        if distance_type == "structural":
            return self._structural_distance(code1, code2)
        elif distance_type == "semantic":
            return self._semantic_distance(code1, code2)
        elif distance_type == "hierarchical":
            return self._hierarchical_distance(code1, code2)
        else:  # weighted
            return self._weighted_distance(code1, code2)
    
    def find_similar_elements(self, target_code: ElementCode, all_codes: List[ElementCode], 
                            threshold: float = 0.3, max_results: int = 10) -> List[Tuple[ElementCode, float]]:
        """Find elements similar to the target"""
        
        distances = []
        for code in all_codes:
            if code.element_id != target_code.element_id:
                dist = self.calculate_distance(target_code, code)
                distances.append((code, dist))
        
        # Sort by distance and filter by threshold
        distances.sort(key=lambda x: x[1])
        return [(code, dist) for code, dist in distances[:max_results] if dist <= threshold]
    
    def cluster_by_hierarchy(self, codes: List[ElementCode]) -> Dict[str, List[ElementCode]]:
        """Cluster elements by hierarchical level and structural similarity"""
        
        clusters = {}
        
        for code in codes:
            # Create cluster key based on hierarchy and structure
            key = f"L{code.hierarchy_level}_{code.structural_signature[:8]}"
            
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(code)
        
        return clusters
    
    # Private helper methods
    
    def _normalize_font_size(self, element: LayoutElement) -> float:
        """Normalize font size to 0-1 scale"""
        if not element.style or not element.style.primary_font or not element.style.primary_font.size:
            return 0.5  # Default middle value
        
        size = element.style.primary_font.size
        min_size = self.font_size_stats["min"]
        max_size = self.font_size_stats["max"]
        
        # Min-max normalization
        normalized = (size - min_size) / (max_size - min_size) if max_size > min_size else 0.5
        return max(0.0, min(1.0, normalized))
    
    def _encode_font_style(self, element: LayoutElement) -> int:
        """Encode font style as bit flags"""
        if not element.style or not element.style.primary_font:
            return 0
        
        font = element.style.primary_font
        code = 0
        
        if font.bold: code |= 1      # Bit 0
        if font.italic: code |= 2    # Bit 1
        if font.underline: code |= 4 # Bit 2
        if font.strikethrough: code |= 8  # Bit 3
        if font.superscript: code |= 16   # Bit 4
        if font.subscript: code |= 32     # Bit 5
        
        return code
    
    def _encode_alignment(self, element: LayoutElement) -> int:
        """Encode text alignment"""
        if not element.style or not element.style.paragraph_format:
            return 0  # Default LEFT
        
        alignment = element.style.paragraph_format.alignment
        alignment_map = {
            TextAlignment.LEFT: 0,
            TextAlignment.CENTER: 1,
            TextAlignment.RIGHT: 2,
            TextAlignment.JUSTIFY: 3,
            TextAlignment.DISTRIBUTE: 4,
            TextAlignment.UNKNOWN: 0
        }
        
        return alignment_map.get(alignment, 0)
    
    def _encode_position(self, element: LayoutElement) -> Tuple[float, float]:
        """Encode normalized position"""
        if not element.bbox:
            return (0.0, 0.0)
        
        # Normalize to document dimensions
        x_norm = element.bbox.x1 / self.position_stats["width"]
        y_norm = element.bbox.y1 / self.position_stats["height"]
        
        return (max(0.0, min(1.0, x_norm)), max(0.0, min(1.0, y_norm)))
    
    def _analyze_hierarchy(self, element: LayoutElement) -> Dict[str, Any]:
        """Analyze hierarchical properties"""
        result = {"level": 0, "path": "", "number_type": None}
        
        if not element.text:
            return result
        
        # Extract title number
        title_number = title_number_extraction(element.text)
        if title_number:
            result["level"] = title_number.level
            result["path"] = title_number.raw_text
            result["number_type"] = title_number.number_type.value
        else:
            # Fallback: estimate level from font size and element type
            if element.element_type in [ElementType.TITLE, ElementType.HEADING]:
                font_size = 12.0
                if element.style and element.style.primary_font and element.style.primary_font.size:
                    font_size = element.style.primary_font.size
                
                # Larger fonts = higher level (lower number)
                if font_size >= 18:
                    result["level"] = 0
                elif font_size >= 16:
                    result["level"] = 1
                elif font_size >= 14:
                    result["level"] = 2
                else:
                    result["level"] = 3
        
        return result
    
    def _analyze_content(self, element: LayoutElement) -> Dict[str, Any]:
        """Analyze content properties"""
        result = {"hash": "", "embedding": None, "length_norm": 0.0}
        
        if not element.text:
            return result
        
        # Content hash for quick comparison
        result["hash"] = hashlib.md5(element.text.encode()).hexdigest()[:16]
        
        # Semantic embedding
        if self.use_semantic_embeddings and self.embedding_model:
            try:
                embedding = self.embedding_model.encode(element.text)
                result["embedding"] = embedding.tolist()
            except Exception:
                pass
        
        # Normalized content length
        length = len(element.text)
        min_len = self.content_length_stats["min"]
        max_len = self.content_length_stats["max"]
        result["length_norm"] = (length - min_len) / (max_len - min_len) if max_len > min_len else 0.5
        result["length_norm"] = max(0.0, min(1.0, result["length_norm"]))
        
        return result
    
    def _encode_element_type(self, element: LayoutElement) -> int:
        """Encode element type as integer"""
        type_map = {
            ElementType.PLAIN_TEXT: 0,
            ElementType.TITLE: 1,
            ElementType.HEADING: 2,
            ElementType.FIGURE: 3,
            ElementType.FIGURE_CAPTION: 4,
            ElementType.TABLE: 5,
            ElementType.TABLE_CAPTION: 6,
            ElementType.HEADER: 7,
            ElementType.FOOTER: 8,
            ElementType.REFERENCE: 9,
            ElementType.EQUATION: 10,
            ElementType.LIST: 11,
            ElementType.PARAGRAPH: 12,
            ElementType.UNKNOWN: 13
        }
        
        return type_map.get(element.element_type, 13)
    
    def _analyze_spacing(self, element: LayoutElement, context_elements: Optional[List[LayoutElement]]) -> Dict[str, float]:
        """Analyze spacing properties"""
        result = {"before": 0.0, "after": 0.0, "indentation": 0.0}
        
        if not element.bbox or not context_elements:
            return result
        
        # Find previous and next elements by position
        current_y = element.bbox.y1
        prev_element = None
        next_element = None
        
        for other in context_elements:
            if other.id == element.id or not other.bbox:
                continue
            
            other_y = other.bbox.y1
            if other_y < current_y and (prev_element is None or other_y > prev_element.bbox.y1):
                prev_element = other
            elif other_y > current_y and (next_element is None or other_y < next_element.bbox.y1):
                next_element = other
        
        # Calculate spacing
        if prev_element:
            spacing_before = current_y - prev_element.bbox.y2
            result["before"] = max(0.0, min(1.0, spacing_before / 50.0))  # Normalize to reasonable range
        
        if next_element:
            spacing_after = next_element.bbox.y1 - element.bbox.y2
            result["after"] = max(0.0, min(1.0, spacing_after / 50.0))
        
        # Indentation (relative to page margin)
        result["indentation"] = max(0.0, min(1.0, element.bbox.x1 / 100.0))  # Normalize to reasonable range
        
        return result
    
    def _generate_structural_signature(self, code: ElementCode) -> str:
        """Generate a compact structural signature for quick matching"""
        # Combine key structural features into a hash
        signature_parts = [
            f"fs{code.font_size_norm:.2f}",
            f"st{code.font_style_code}",
            f"al{code.alignment_code}",
            f"hl{code.hierarchy_level}",
            f"et{code.element_type_code}"
        ]
        
        signature_string = "_".join(signature_parts)
        return hashlib.md5(signature_string.encode()).hexdigest()[:12]
    
    def _weighted_distance(self, code1: ElementCode, code2: ElementCode) -> float:
        """Calculate weighted distance between codes"""
        distances = []
        
        # Font size distance
        fs_dist = abs(code1.font_size_norm - code2.font_size_norm)
        distances.append(fs_dist * self.weights.font_size)
        
        # Font style distance (Hamming distance on bit flags)
        style_dist = bin(code1.font_style_code ^ code2.font_style_code).count('1') / 6.0  # Max 6 bits
        distances.append(style_dist * self.weights.font_style)
        
        # Alignment distance
        align_dist = 1.0 if code1.alignment_code != code2.alignment_code else 0.0
        distances.append(align_dist * self.weights.alignment)
        
        # Position distance
        pos_dist = math.sqrt((code1.position_code[0] - code2.position_code[0])**2 + 
                           (code1.position_code[1] - code2.position_code[1])**2)
        distances.append(pos_dist * self.weights.position)
        
        # Hierarchy distance
        hier_dist = abs(code1.hierarchy_level - code2.hierarchy_level) / 10.0  # Normalize
        distances.append(hier_dist * self.weights.hierarchy)
        
        # Element type distance
        type_dist = 1.0 if code1.element_type_code != code2.element_type_code else 0.0
        distances.append(type_dist * self.weights.element_type)
        
        # Semantic distance (if available)
        if (code1.content_embedding and code2.content_embedding and 
            len(code1.content_embedding) == len(code2.content_embedding)):
            
            # Cosine distance
            vec1 = np.array(code1.content_embedding)
            vec2 = np.array(code2.content_embedding)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = dot_product / (norm1 * norm2)
                semantic_dist = 1.0 - cosine_sim
                distances.append(semantic_dist * self.weights.content_semantic)
        
        # Content length distance
        length_dist = abs(code1.content_length_norm - code2.content_length_norm)
        distances.append(length_dist * 0.3)  # Lower weight
        
        # Spacing distance
        spacing_dist = (abs(code1.spacing_before - code2.spacing_before) + 
                       abs(code1.spacing_after - code2.spacing_after) + 
                       abs(code1.indentation - code2.indentation)) / 3.0
        distances.append(spacing_dist * self.weights.spacing)
        
        # Weighted average
        total_weight = sum([self.weights.font_size, self.weights.font_style, self.weights.alignment,
                           self.weights.position, self.weights.hierarchy, self.weights.element_type,
                           self.weights.content_semantic, 0.3, self.weights.spacing])
        
        return sum(distances) / total_weight
    
    def _structural_distance(self, code1: ElementCode, code2: ElementCode) -> float:
        """Calculate distance based only on structural features (no semantic)"""
        if code1.structural_signature == code2.structural_signature:
            return 0.0
        
        # Focus on structural elements
        distances = []
        
        distances.append(abs(code1.font_size_norm - code2.font_size_norm))
        distances.append(bin(code1.font_style_code ^ code2.font_style_code).count('1') / 6.0)
        distances.append(1.0 if code1.alignment_code != code2.alignment_code else 0.0)
        distances.append(abs(code1.hierarchy_level - code2.hierarchy_level) / 10.0)
        distances.append(1.0 if code1.element_type_code != code2.element_type_code else 0.0)
        
        return sum(distances) / len(distances)
    
    def _semantic_distance(self, code1: ElementCode, code2: ElementCode) -> float:
        """Calculate distance based primarily on semantic content"""
        if not (code1.content_embedding and code2.content_embedding):
            # Fallback to content hash comparison
            return 0.0 if code1.content_hash == code2.content_hash else 1.0
        
        # Cosine distance
        vec1 = np.array(code1.content_embedding)
        vec2 = np.array(code2.content_embedding)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            return 1.0 - cosine_sim
        
        return 1.0
    
    def _hierarchical_distance(self, code1: ElementCode, code2: ElementCode) -> float:
        """Calculate distance based on hierarchical relationship"""
        level_diff = abs(code1.hierarchy_level - code2.hierarchy_level)
        
        # Same level = potentially siblings
        if level_diff == 0:
            return 0.1  # Small distance for same level
        
        # Adjacent levels = potentially parent-child
        if level_diff == 1:
            return 0.3
        
        # Further apart = less related
        return min(1.0, level_diff / 5.0) 