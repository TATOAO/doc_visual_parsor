#!/usr/bin/env python3
"""
Demonstration script for the Section Tree Builder.
Shows how to construct hierarchical section trees from layout elements using the element encoder.
"""

import json
import asyncio
from pathlib import Path

from models.schemas.layout_schemas import LayoutExtractionResult
from .section_tree_builder import SectionTreeBuilder, TreeBuildingConfig
from .element_encoder import ElementEncoder, FeatureWeights


async def demo_section_tree():
    """Demonstrate the section tree building process"""
    
    print("=== Section Tree Builder Demo ===\n")
    
    # Load sample document
    sample_file = Path("hybrid_extraction_result.json")
    if not sample_file.exists():
        print(f"Sample file {sample_file} not found. Please run the layout extraction first.")
        return
    
    print("Loading sample document...")
    with open(sample_file, "r") as f:
        layout_data = json.load(f)
    
    layout_result = LayoutExtractionResult(
        elements=layout_data["elements"], 
        metadata=layout_data["metadata"]
    )
    
    print(f"Loaded {len(layout_result.elements)} elements\n")
    
    # Configure different tree building strategies
    print("=== Configuration Options ===")
    
    # 1. Hierarchy-focused configuration
    print("1. Hierarchy-focused configuration:")
    hierarchy_weights = FeatureWeights(
        font_size=1.2,
        font_style=0.9,
        alignment=0.7,
        position=0.5,
        hierarchy=2.0,  # Strong emphasis on hierarchy
        content_semantic=0.6,
        element_type=1.1,
        spacing=0.8
    )
    
    hierarchy_config = TreeBuildingConfig(
        merge_distance_threshold=0.1,  # More aggressive merging
        max_hierarchy_gap=2,
        min_content_length=10
    )
    
    # 2. Structure-focused configuration
    print("2. Structure-focused configuration:")
    structure_weights = FeatureWeights(
        font_size=1.5,  # Emphasize visual structure
        font_style=1.3,
        alignment=1.0,
        position=0.7,
        hierarchy=1.0,
        content_semantic=0.4,
        element_type=1.2,
        spacing=1.0
    )
    
    structure_config = TreeBuildingConfig(
        merge_distance_threshold=0.15,
        max_hierarchy_gap=1,  # Stricter hierarchy
        min_content_length=5
    )
    
    # Build trees with different configurations
    await build_and_analyze_tree(
        "Hierarchy-focused", 
        layout_result.elements, 
        hierarchy_weights, 
        hierarchy_config
    )
    
    print("\n" + "="*60 + "\n")
    
    await build_and_analyze_tree(
        "Structure-focused", 
        layout_result.elements, 
        structure_weights, 
        structure_config
    )


async def build_and_analyze_tree(name: str, elements, weights: FeatureWeights, config: TreeBuildingConfig):
    """Build and analyze a section tree with given configuration"""
    
    print(f"=== {name} Tree Building ===\n")
    
    # Initialize encoder and tree builder
    encoder = ElementEncoder(weights=weights, use_semantic_embeddings=False)
    tree_builder = SectionTreeBuilder(encoder=encoder, config=config)
    
    # Build the tree
    print("Building section tree...")
    root_section = await tree_builder.build_tree(elements)
    
    # Print tree structure
    print(f"\n=== {name} Tree Structure ===")
    tree_builder.print_tree(root_section)
    
    # Get and display statistics
    print(f"\n=== {name} Tree Statistics ===")
    stats = tree_builder.get_tree_statistics(root_section)
    
    print(f"Total sections: {stats['total_sections']}")
    print(f"Maximum depth: {stats['max_depth']}")
    print(f"Average content length: {stats['avg_content_length']:.1f} characters")
    print(f"Sections with children: {stats['sections_with_children']}")
    print(f"Leaf sections: {stats['leaf_sections']}")
    
    print("\nLevel distribution:")
    for level, count in sorted(stats['level_distribution'].items()):
        print(f"  Level {level}: {count} sections")
    
    # Export tree
    output_file = f"section_tree_{name.lower().replace('-', '_')}.json"
    tree_builder.export_tree(root_section, output_file)
    
    # Analyze section quality
    print(f"\n=== {name} Section Quality Analysis ===")
    await analyze_section_quality(root_section, tree_builder)
    
    return root_section


async def analyze_section_quality(root_section, tree_builder):
    """Analyze the quality of the generated section tree"""
    
    all_sections = [root_section] + tree_builder._get_all_descendants(root_section)
    
    # Filter out root section for analysis
    content_sections = [s for s in all_sections if s.level >= 0]
    
    if not content_sections:
        print("No content sections found.")
        return
    
    # Title analysis
    titled_sections = [s for s in content_sections if s.title and s.title != "Untitled Section"]
    print(f"Sections with meaningful titles: {len(titled_sections)}/{len(content_sections)} ({len(titled_sections)/len(content_sections)*100:.1f}%)")
    
    # Content analysis
    content_sections_with_text = [s for s in content_sections if s.content.strip()]
    print(f"Sections with content: {len(content_sections_with_text)}/{len(content_sections)} ({len(content_sections_with_text)/len(content_sections)*100:.1f}%)")
    
    # Hierarchy analysis
    if content_sections:
        max_level = max(s.level for s in content_sections)
        min_level = min(s.level for s in content_sections)
        print(f"Hierarchy depth: {max_level - min_level + 1} levels (from {min_level} to {max_level})")
    
    # Content distribution
    content_lengths = [len(s.content) for s in content_sections if s.content]
    if content_lengths:
        avg_length = sum(content_lengths) / len(content_lengths)
        max_length = max(content_lengths)
        min_length = min(content_lengths)
        print(f"Content length distribution: avg={avg_length:.1f}, min={min_length}, max={max_length}")
    
    # Show sample sections
    print(f"\nSample sections:")
    sample_sections = content_sections[:5]  # First 5 sections
    for i, section in enumerate(sample_sections):
        title_preview = section.title[:40] + "..." if len(section.title) > 40 else section.title
        content_preview = section.content[:60] + "..." if len(section.content) > 60 else section.content
        
        metadata = getattr(section, '_metadata', {})
        hierarchy_path = metadata.get('hierarchy_path', 'N/A')
        
        print(f"  {i+1}. L{section.level}: {title_preview}")
        print(f"     Content: {content_preview}")
        print(f"     Hierarchy: {hierarchy_path}")
        print(f"     Children: {len(section.sub_sections)}")
        print()


def validate_tree_structure(root_section):
    """Validate the structural integrity of the section tree"""
    
    print("=== Tree Structure Validation ===")
    
    errors = []
    warnings = []
    
    def validate_node(section, parent=None, visited=None):
        if visited is None:
            visited = set()
        
        # Check for circular references
        if id(section) in visited:
            errors.append(f"Circular reference detected in section: {section.title}")
            return
        
        visited.add(id(section))
        
        # Check parent-child consistency
        if parent and section.parent_section != parent:
            errors.append(f"Parent-child inconsistency in section: {section.title}")
        
        # Check level consistency
        if parent and section.level <= parent.level:
            warnings.append(f"Level inconsistency: child level {section.level} <= parent level {parent.level} in: {section.title}")
        
        # Check for empty critical fields
        if not section.title or section.title == "Untitled Section":
            warnings.append(f"Section without meaningful title at level {section.level}")
        
        # Recursively validate children
        for child in section.sub_sections:
            validate_node(child, section, visited.copy())
    
    validate_node(root_section)
    
    # Report results
    if not errors and not warnings:
        print("✅ Tree structure is valid with no issues.")
    else:
        if errors:
            print(f"❌ Found {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
        
        if warnings:
            print(f"⚠️  Found {len(warnings)} warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    
    return len(errors) == 0


async def compare_tree_configurations():
    """Compare different tree building configurations side by side"""
    
    print("=== Configuration Comparison ===\n")
    
    # Load sample document
    sample_file = Path("hybrid_extraction_result.json")
    if not sample_file.exists():
        print("Sample file not found for comparison.")
        return
    
    with open(sample_file, "r") as f:
        layout_data = json.load(f)
    
    layout_result = LayoutExtractionResult(
        elements=layout_data["elements"], 
        metadata=layout_data["metadata"]
    )
    
    # Define different configurations
    configs = {
        "Conservative": {
            "weights": FeatureWeights(hierarchy=1.0, font_size=1.0, merge_distance_threshold=0.05),
            "config": TreeBuildingConfig(merge_distance_threshold=0.05, min_content_length=15)
        },
        "Balanced": {
            "weights": FeatureWeights(hierarchy=1.2, font_size=1.1),
            "config": TreeBuildingConfig(merge_distance_threshold=0.15, min_content_length=5)
        },
        "Aggressive": {
            "weights": FeatureWeights(hierarchy=1.5, font_size=1.3),
            "config": TreeBuildingConfig(merge_distance_threshold=0.25, min_content_length=3)
        }
    }
    
    results = {}
    
    # Build trees with each configuration
    for name, cfg in configs.items():
        encoder = ElementEncoder(weights=cfg["weights"], use_semantic_embeddings=False)
        tree_builder = SectionTreeBuilder(encoder=encoder, config=cfg["config"])
        
        print(f"Building {name} tree...")
        root_section = await tree_builder.build_tree(layout_result.elements)
        stats = tree_builder.get_tree_statistics(root_section)
        
        results[name] = {
            "tree": root_section,
            "stats": stats,
            "builder": tree_builder
        }
    
    # Compare results
    print("\n=== Configuration Comparison Results ===")
    print(f"{'Metric':<25} {'Conservative':<12} {'Balanced':<12} {'Aggressive':<12}")
    print("-" * 65)
    
    metrics = [
        ("Total sections", "total_sections"),
        ("Max depth", "max_depth"),
        ("Avg content length", "avg_content_length"),
        ("Leaf sections", "leaf_sections"),
        ("Sections w/ children", "sections_with_children")
    ]
    
    for metric_name, metric_key in metrics:
        row = f"{metric_name:<25}"
        for config_name in ["Conservative", "Balanced", "Aggressive"]:
            value = results[config_name]["stats"][metric_key]
            if isinstance(value, float):
                row += f"{value:<12.1f}"
            else:
                row += f"{value:<12}"
        print(row)
    
    # Recommend best configuration
    print(f"\n=== Recommendations ===")
    
    # Simple scoring based on balance of metrics
    scores = {}
    for config_name, result in results.items():
        stats = result["stats"]
        # Score based on reasonable section count, depth, and content distribution
        score = 0
        
        # Prefer moderate section counts (not too few, not too many)
        section_count = stats["total_sections"]
        if 5 <= section_count <= 20:
            score += 2
        elif section_count > 20:
            score += 1
        
        # Prefer reasonable depth
        depth = stats["max_depth"]
        if 2 <= depth <= 5:
            score += 2
        elif depth > 5:
            score += 1
        
        # Prefer sections with content
        if stats["leaf_sections"] > 0:
            score += 1
        
        scores[config_name] = score
    
    best_config = max(scores.keys(), key=lambda k: scores[k])
    print(f"Recommended configuration: {best_config}")
    print(f"Scores: {scores}")


# python -m models.layout_structuring.content_merger.demo_section_tree
if __name__ == "__main__":
    asyncio.run(demo_section_tree()) 