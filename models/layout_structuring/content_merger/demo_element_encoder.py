#!/usr/bin/env python3
"""
Demonstration script for the ElementEncoder system.
Shows how to use compact element representations for document structure analysis.
"""

import json
import asyncio
from typing import List, Dict
from pathlib import Path

from models.schemas.layout_schemas import LayoutExtractionResult, LayoutElement
from .element_encoder import ElementEncoder, FeatureWeights, ElementCode


async def demo_element_encoding():
    """Demonstrate the element encoding and analysis system"""
    
    print("=== Element Encoder Demo ===\n")
    
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
    
    # Initialize encoder with different weight configurations
    print("1. Standard Configuration:")
    standard_encoder = ElementEncoder(use_semantic_embeddings=False)  # Disable for demo
    
    print("2. Hierarchy-focused Configuration:")
    hierarchy_weights = FeatureWeights(
        font_size=1.2,
        font_style=0.9,
        alignment=0.7,
        position=0.5,
        hierarchy=2.0,  # Higher weight on hierarchy
        content_semantic=0.6,
        element_type=1.1,
        spacing=0.8
    )
    hierarchy_encoder = ElementEncoder(weights=hierarchy_weights, use_semantic_embeddings=False)
    
    print("3. Style-focused Configuration:")
    style_weights = FeatureWeights(
        font_size=1.5,  # Higher weight on visual features
        font_style=1.3,
        alignment=1.0,
        position=0.6,
        hierarchy=0.8,
        content_semantic=0.5,
        element_type=0.9,
        spacing=0.4
    )
    style_encoder = ElementEncoder(weights=style_weights, use_semantic_embeddings=False)
    
    # Encode elements
    print("\nEncoding elements...")
    standard_codes = standard_encoder.encode_elements(layout_result.elements)
    hierarchy_codes = hierarchy_encoder.encode_elements(layout_result.elements)
    style_codes = style_encoder.encode_elements(layout_result.elements)
    
    print(f"Generated {len(standard_codes)} element codes\n")
    
    # Analyze first few elements
    print("=== Element Analysis ===")
    analyze_elements(layout_result.elements[:10], standard_codes[:10])
    
    # Demonstrate similarity analysis
    print("\n=== Similarity Analysis ===")
    demonstrate_similarity_analysis(layout_result.elements, standard_codes)
    
    # Demonstrate clustering
    print("\n=== Hierarchical Clustering ===")
    demonstrate_clustering(layout_result.elements, hierarchy_codes)
    
    # Compare different encodings
    print("\n=== Encoding Comparison ===")
    compare_encodings(layout_result.elements, standard_codes, hierarchy_codes, style_codes)
    
    # Generate analysis report
    print("\n=== Analysis Report ===")
    generate_analysis_report(layout_result.elements, standard_codes, standard_encoder)


def analyze_elements(elements: List[LayoutElement], codes: List[ElementCode]):
    """Analyze individual elements and their codes"""
    
    print("Element Code Analysis:")
    print("-" * 80)
    print(f"{'ID':<4} {'Type':<12} {'Font':<6} {'Style':<6} {'Level':<5} {'Signature':<12} {'Text':<30}")
    print("-" * 80)
    
    for element, code in zip(elements[:10], codes[:10]):
        text_preview = (element.text or "")[:30].replace('\n', ' ')
        
        print(f"{code.element_id:<4} "
              f"{element.element_type.value:<12} "
              f"{code.font_size_norm:.2f}   "
              f"{code.font_style_code:<6} "
              f"{code.hierarchy_level:<5} "
              f"{code.structural_signature[:12]:<12} "
              f"{text_preview:<30}")


def demonstrate_similarity_analysis(elements: List[LayoutElement], codes: List[ElementCode]):
    """Demonstrate finding similar elements"""
    
    encoder = ElementEncoder(use_semantic_embeddings=False)
    
    # Find titles/headings
    title_codes = [code for code, elem in zip(codes, elements) 
                   if elem.element_type.value in ['Title', 'Heading']]
    
    if len(title_codes) >= 2:
        print(f"Analyzing similarity between title elements...")
        
        target_code = title_codes[0]
        target_element = next(elem for elem in elements if elem.id == target_code.element_id)
        
        print(f"\nTarget element (ID: {target_code.element_id}):")
        print(f"  Text: {(target_element.text or '')[:50]}...")
        print(f"  Type: {target_element.element_type.value}")
        print(f"  Hierarchy Level: {target_code.hierarchy_level}")
        print(f"  Font Size (norm): {target_code.font_size_norm:.3f}")
        print(f"  Style Code: {target_code.font_style_code}")
        
        # Find similar elements
        similar_elements = encoder.find_similar_elements(target_code, codes, threshold=0.5, max_results=5)
        
        print(f"\nSimilar elements (distance < 0.5):")
        for similar_code, distance in similar_elements:
            similar_element = next(elem for elem in elements if elem.id == similar_code.element_id)
            print(f"  ID: {similar_code.element_id}, Distance: {distance:.3f}")
            print(f"    Text: {(similar_element.text or '')[:40]}...")
            print(f"    Level: {similar_code.hierarchy_level}, Style: {similar_code.font_style_code}")
    else:
        print("Not enough title elements for similarity analysis")


def demonstrate_clustering(elements: List[LayoutElement], codes: List[ElementCode]):
    """Demonstrate hierarchical clustering"""
    
    encoder = ElementEncoder(use_semantic_embeddings=False)
    clusters = encoder.cluster_by_hierarchy(codes)
    
    print(f"Found {len(clusters)} structural clusters:")
    print("-" * 60)
    
    for cluster_key, cluster_codes in sorted(clusters.items()):
        if len(cluster_codes) > 1:  # Only show clusters with multiple elements
            print(f"\nCluster: {cluster_key} ({len(cluster_codes)} elements)")
            
            for code in cluster_codes[:10]:  # Show first 3 elements
                element = next(elem for elem in elements if elem.id == code.element_id)
                text_preview = (element.text or "")[:40].replace('\n', ' ')
                print(f"  ID: {code.element_id} | {text_preview}")
            
            if len(cluster_codes) > 10:
                print(f"  ... and {len(cluster_codes) - 10} more elements")


def compare_encodings(elements: List[LayoutElement], 
                     standard_codes: List[ElementCode], 
                     hierarchy_codes: List[ElementCode], 
                     style_codes: List[ElementCode]):
    """Compare different encoding strategies"""
    
    print("Encoding Strategy Comparison:")
    print("-" * 70)
    
    # Compare clustering results
    standard_encoder = ElementEncoder(use_semantic_embeddings=False)
    hierarchy_encoder = ElementEncoder(use_semantic_embeddings=False)
    style_encoder = ElementEncoder(use_semantic_embeddings=False)
    
    standard_clusters = standard_encoder.cluster_by_hierarchy(standard_codes)
    hierarchy_clusters = hierarchy_encoder.cluster_by_hierarchy(hierarchy_codes)
    style_clusters = style_encoder.cluster_by_hierarchy(style_codes)
    
    print(f"Standard encoding: {len(standard_clusters)} clusters")
    print(f"Hierarchy-focused: {len(hierarchy_clusters)} clusters")
    print(f"Style-focused: {len(style_clusters)} clusters")
    
    # Compare distance calculations for title elements
    title_elements = [(elem, code) for elem, code in zip(elements, standard_codes) 
                      if elem.element_type.value in ['Title', 'Heading']]
    
    if len(title_elements) >= 2:
        elem1, code1 = title_elements[0]
        elem2, code2 = title_elements[1]
        
        standard_dist = standard_encoder.calculate_distance(code1, code2, "weighted")
        hierarchy_dist = hierarchy_encoder.calculate_distance(
            hierarchy_codes[code1.element_id], hierarchy_codes[code2.element_id], "hierarchical")
        style_dist = style_encoder.calculate_distance(
            style_codes[code1.element_id], style_codes[code2.element_id], "structural")
        
        print(f"\nDistance between first two titles:")
        print(f"  Standard weighted: {standard_dist:.3f}")
        print(f"  Hierarchical: {hierarchy_dist:.3f}")
        print(f"  Structural: {style_dist:.3f}")


def generate_analysis_report(elements: List[LayoutElement], codes: List[ElementCode], encoder: ElementEncoder):
    """Generate a comprehensive analysis report"""
    
    print("Document Structure Analysis Report:")
    print("=" * 50)
    
    # Element type distribution
    type_counts = {}
    for element in elements:
        type_name = element.element_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    print("\nElement Type Distribution:")
    for elem_type, count in sorted(type_counts.items()):
        print(f"  {elem_type}: {count}")
    
    # Hierarchy level distribution
    level_counts = {}
    for code in codes:
        level = code.hierarchy_level
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print("\nHierarchy Level Distribution:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} elements")
    
    # Font size analysis
    font_sizes = [code.font_size_norm for code in codes if code.font_size_norm > 0]
    if font_sizes:
        import numpy as np
        print(f"\nFont Size Analysis (normalized):")
        print(f"  Min: {np.min(font_sizes):.3f}")
        print(f"  Max: {np.max(font_sizes):.3f}")
        print(f"  Mean: {np.mean(font_sizes):.3f}")
        print(f"  Std: {np.std(font_sizes):.3f}")
    
    # Style analysis
    style_counts = {}
    for code in codes:
        style_counts[code.font_style_code] = style_counts.get(code.font_style_code, 0) + 1
    
    print(f"\nStyle Code Distribution:")
    style_names = {0: "Normal", 1: "Bold", 2: "Italic", 3: "Bold+Italic", 
                   4: "Underline", 5: "Bold+Underline"}
    
    for style_code in sorted(style_counts.keys()):
        style_name = style_names.get(style_code, f"Code {style_code}")
        print(f"  {style_name}: {style_counts[style_code]} elements")


# python -m models.layout_structuring.content_merger.demo_element_encoder
if __name__ == "__main__":
    asyncio.run(demo_element_encoding()) 