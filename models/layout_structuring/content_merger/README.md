# Element Encoder System for Document Structure Analysis

This module provides a compact representation system for layout elements that enables distance-based analysis and hierarchical structuring of documents. The system encodes visual, structural, and semantic features into a unified representation that can be used for clustering, similarity analysis, and automated document structuring.

## Overview

The Element Encoder creates a "code" representation for each layout element that captures:

1. **Font Properties**: Size, style (bold, italic, etc.), and formatting
2. **Alignment**: Text alignment and positioning
3. **Hierarchical Information**: Detected hierarchy levels and numbering patterns
4. **Spatial Properties**: Position, spacing, and indentation
5. **Content Features**: Text length, content hash, and optional semantic embeddings
6. **Element Type**: Document element classification

## Key Components

### ElementCode

A compact representation of a layout element containing:

```python
class ElementCode(BaseModel):
    element_id: int
    font_size_norm: float        # Normalized font size (0-1)
    font_style_code: int         # Bit flags for style properties
    alignment_code: int          # Encoded alignment
    position_code: Tuple[float, float]  # Normalized position
    hierarchy_level: int         # Detected hierarchy level
    hierarchy_path: str          # Hierarchy path string
    content_hash: str           # Content hash for comparison
    content_embedding: List[float]  # Optional semantic embedding
    structural_signature: str    # Quick structural matching
    # ... additional properties
```

### ElementEncoder

The main encoder class that:
- Converts layout elements to compact codes
- Calculates distances between elements
- Provides clustering and similarity analysis
- Supports configurable feature weights

### FeatureWeights

Configurable weights for different aspects of the analysis:

```python
@dataclass
class FeatureWeights:
    font_size: float = 1.0
    font_style: float = 0.8
    alignment: float = 0.6
    position: float = 0.7
    hierarchy: float = 1.2      # Higher weight for hierarchy
    content_semantic: float = 0.9
    element_type: float = 1.0
    spacing: float = 0.5
```

## Usage Examples

### Basic Usage

```python
from .element_encoder import ElementEncoder, FeatureWeights
from models.schemas.layout_schemas import LayoutExtractionResult

# Initialize encoder
encoder = ElementEncoder(use_semantic_embeddings=False)

# Load your document elements
layout_result = LayoutExtractionResult(elements=elements, metadata=metadata)

# Encode all elements
element_codes = encoder.encode_elements(layout_result.elements)

# Analyze similarity between elements
code1, code2 = element_codes[0], element_codes[1]
distance = encoder.calculate_distance(code1, code2)
print(f"Distance between elements: {distance:.3f}")
```

### Custom Weight Configuration

```python
# Hierarchy-focused analysis
hierarchy_weights = FeatureWeights(
    font_size=1.2,
    font_style=0.9,
    hierarchy=2.0,    # Emphasize hierarchical features
    content_semantic=0.6,
    element_type=1.1
)

encoder = ElementEncoder(weights=hierarchy_weights)
```

### Clustering and Analysis

```python
# Find similar elements
target_code = element_codes[0]
similar_elements = encoder.find_similar_elements(
    target_code, element_codes, 
    threshold=0.3, max_results=5
)

# Cluster by hierarchy
clusters = encoder.cluster_by_hierarchy(element_codes)
print(f"Found {len(clusters)} structural clusters")
```

### Integration with ContentMerger

```python
from . import ContentMerger

# Use with custom weights
content_merger = ContentMerger(
    use_element_encoder=True,
    encoder_weights=hierarchy_weights
)

# Process elements
processed_elements = await content_merger.construct_section(elements)

# Get analysis report
report = await content_merger.get_element_analysis_report(elements)
```

## Distance Metrics

The system supports multiple distance calculation methods:

### 1. Weighted Distance (Default)
Combines all features with configurable weights:
- Font size difference
- Style bit flag differences (Hamming distance)
- Alignment differences
- Position Euclidean distance
- Hierarchy level differences
- Semantic similarity (cosine distance)
- Content length differences
- Spacing differences

### 2. Structural Distance
Focuses only on visual/structural features:
- Font properties
- Alignment
- Hierarchy level
- Element type
- Excludes semantic content

### 3. Semantic Distance
Emphasizes content similarity:
- Semantic embeddings (if available)
- Content hash comparison
- Text length similarity

### 4. Hierarchical Distance
Specialized for hierarchy analysis:
- Same level → low distance (potential siblings)
- Adjacent levels → medium distance (parent-child)
- Distant levels → high distance

## Feature Encoding Details

### Font Style Encoding
Uses bit flags for efficient storage:
```
Bit 0: Bold
Bit 1: Italic  
Bit 2: Underline
Bit 3: Strikethrough
Bit 4: Superscript
Bit 5: Subscript
```

### Alignment Encoding
```
0: LEFT
1: CENTER
2: RIGHT
3: JUSTIFY
4: DISTRIBUTE
```

### Hierarchy Analysis
Combines multiple approaches:
1. **Title Number Extraction**: Detects patterns like "第一条", "(1)", "1.1", etc.
2. **Font Size Analysis**: Larger fonts typically indicate higher hierarchy
3. **Element Type**: TITLE/HEADING elements get priority
4. **Structural Patterns**: Consistent formatting suggests same level

### Normalization
All continuous features are normalized to [0,1] range:
- Font sizes normalized by document statistics
- Positions normalized by page dimensions
- Content lengths normalized by document statistics

## Performance Considerations

### Memory Usage
- Base ElementCode: ~500 bytes per element
- With embeddings: +2KB per element (384-dim embeddings)
- Recommended: Disable embeddings for large documents

### Speed Optimization
- Structural signatures enable fast pre-filtering
- Bit operations for style comparisons
- Vectorized distance calculations

### Scalability
- Linear time complexity for encoding
- O(n²) for pairwise distance calculations
- Clustering: O(n log n) average case

## Configuration Recommendations

### For Legal Documents
```python
legal_weights = FeatureWeights(
    hierarchy=2.0,     # Strong emphasis on numbering
    font_size=1.5,     # Font size important for structure
    font_style=1.2,    # Bold/italic significant
    alignment=0.8,
    position=0.6,
    content_semantic=0.4,  # Less emphasis on content
    spacing=0.9        # Spacing patterns important
)
```

### For Academic Papers
```python
academic_weights = FeatureWeights(
    content_semantic=1.5,  # Content similarity important
    hierarchy=1.3,         # Section structure
    font_size=1.0,
    font_style=0.8,
    element_type=1.2,      # Figure/table captions, etc.
    position=0.5
)
```

### For Layout Analysis
```python
layout_weights = FeatureWeights(
    position=1.5,      # Spatial relationships key
    spacing=1.3,       # Layout spacing patterns
    alignment=1.2,     # Text alignment important
    font_size=1.0,
    hierarchy=0.8,
    content_semantic=0.3
)
```

## Testing and Validation

Run the demonstration script:
```bash
cd models/layout_structuring/content_merger
python demo_element_encoder.py
```

This will:
1. Load sample document
2. Encode elements with different configurations
3. Perform similarity analysis
4. Generate clustering results
5. Compare different encoding strategies
6. Output comprehensive analysis report

## Extending the System

### Adding New Features
1. Extend `ElementCode` with new properties
2. Implement encoding logic in `ElementEncoder._encode_*` methods
3. Update distance calculation in `_weighted_distance`
4. Add corresponding weight in `FeatureWeights`

### Custom Distance Metrics
Implement new distance functions in `ElementEncoder`:
```python
def _custom_distance(self, code1: ElementCode, code2: ElementCode) -> float:
    # Your custom logic here
    return distance_value
```

### Integration with ML Models
The compact codes can be used as features for:
- Document classification
- Structure prediction
- Similarity learning
- Clustering algorithms

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Disable semantic embeddings
2. **Slow Performance**: Use structural distance for large documents
3. **Poor Clustering**: Adjust feature weights for your document type
4. **Missing Dependencies**: Install optional packages:
   ```bash
   uv pip install sentence-transformers numpy
   ```

### Debug Mode
Enable verbose output:
```python
encoder = ElementEncoder(use_semantic_embeddings=False)
# Encoder will print analysis steps
```

## Future Enhancements

- [ ] Support for table structure encoding
- [ ] Image/figure relationship analysis  
- [ ] Multi-language number pattern detection
- [ ] Adaptive weight learning
- [ ] GPU acceleration for embeddings
- [ ] Integration with transformer models 