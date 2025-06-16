#!/usr/bin/env python3
"""
Complete Pipeline Demo: From Layout Elements to Section Trees
Shows the full process of document structure reconstruction using element encoding and tree building.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any

from models.schemas.layout_schemas import LayoutExtractionResult
from models.schemas.schemas import Section
from . import ContentMerger
from .element_encoder import FeatureWeights
from .section_tree_builder import TreeBuildingConfig


async def main():
    """Main demonstration of the complete pipeline"""
    
    print("🚀 Complete Document Structure Reconstruction Pipeline Demo")
    print("=" * 70)
    
    # Load sample document
    sample_file = Path("hybrid_extraction_result.json")
    if not sample_file.exists():
        print(f"❌ Sample file {sample_file} not found.")
        print("Please run the layout extraction first to generate hybrid_extraction_result.json")
        return
    
    print("📄 Loading document...")
    with open(sample_file, "r") as f:
        layout_data = json.load(f)
    
    layout_result = LayoutExtractionResult(
        elements=layout_data["elements"], 
        metadata=layout_data["metadata"]
    )
    
    print(f"✅ Loaded {len(layout_result.elements)} layout elements\n")
    
    # Show element overview
    await show_element_overview(layout_result.elements)
    
    # Demo different configurations
    configurations = {
        "🏛️ Legal Document Focus": {
            "description": "Optimized for legal documents with strong hierarchy emphasis",
            "weights": FeatureWeights(
                hierarchy=2.0,      # Strong emphasis on numbering
                font_size=1.5,      # Font size important for structure
                font_style=1.2,     # Bold/italic significant
                alignment=0.8,
                position=0.6,
                content_semantic=0.4,
                element_type=1.1,
                spacing=0.9
            ),
            "config": TreeBuildingConfig(
                merge_distance_threshold=0.1,
                max_hierarchy_gap=2,
                min_content_length=8
            )
        },
        
        "📚 Academic Paper Focus": {
            "description": "Optimized for academic papers with content emphasis",
            "weights": FeatureWeights(
                content_semantic=1.5,  # Content similarity important
                hierarchy=1.3,         # Section structure
                font_size=1.0,
                font_style=0.8,
                element_type=1.2,      # Figure/table captions
                position=0.5,
                alignment=0.7,
                spacing=0.6
            ),
            "config": TreeBuildingConfig(
                merge_distance_threshold=0.15,
                max_hierarchy_gap=2,
                min_content_length=10
            )
        },
        
        "📋 General Document": {
            "description": "Balanced approach for general documents",
            "weights": FeatureWeights(
                hierarchy=1.2,
                font_size=1.1,
                font_style=0.9,
                alignment=0.7,
                position=0.6,
                content_semantic=0.8,
                element_type=1.0,
                spacing=0.7
            ),
            "config": TreeBuildingConfig(
                merge_distance_threshold=0.15,
                max_hierarchy_gap=2,
                min_content_length=5
            )
        }
    }
    
    # Process with each configuration
    results = {}
    
    for config_name, config_data in configurations.items():
        print(f"\n{config_name}")
        print("─" * 50)
        print(f"Description: {config_data['description']}")
        
        # Initialize ContentMerger with specific configuration
        content_merger = ContentMerger(
            use_element_encoder=True,
            encoder_weights=config_data['weights'],
            tree_config=config_data['config']
        )
        
        # Build section tree
        print("🔧 Building section tree...")
        root_section = await content_merger.construct_section_tree(layout_result.elements)
        
        # Analyze results
        tree_report = await content_merger.get_tree_analysis_report(root_section)
        element_report = await content_merger.get_element_analysis_report(layout_result.elements)
        
        results[config_name] = {
            'root_section': root_section,
            'tree_report': tree_report,
            'element_report': element_report,
            'merger': content_merger
        }
        
        # Show results summary
        await show_results_summary(config_name, root_section, tree_report)
        
        # Export tree
        safe_name = config_name.replace("🏛️", "legal").replace("📚", "academic").replace("📋", "general").replace(" ", "_").lower()
        output_file = f"section_tree_{safe_name}.json"
        content_merger.tree_builder.export_tree(root_section, output_file)
        print(f"💾 Exported tree to: {output_file}")
    
    # Compare configurations
    print(f"\n🔍 CONFIGURATION COMPARISON")
    print("=" * 70)
    await compare_configurations(results)
    
    # Demonstrate tree traversal and analysis
    print(f"\n🌳 DETAILED TREE ANALYSIS")
    print("=" * 70)
    
    # Pick the best-performing configuration for detailed analysis
    best_config = pick_best_configuration(results)
    await detailed_tree_analysis(results[best_config])
    
    # Show practical applications
    print(f"\n💡 PRACTICAL APPLICATIONS")
    print("=" * 70)
    await demonstrate_applications(results[best_config])


async def show_element_overview(elements):
    """Show overview of the input elements"""
    
    print("📊 ELEMENT OVERVIEW")
    print("─" * 30)
    
    # Element type distribution
    type_counts = {}
    for element in elements:
        type_name = element.element_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    print("Element Types:")
    for elem_type, count in sorted(type_counts.items()):
        print(f"  📌 {elem_type}: {count}")
    
    # Text analysis
    text_elements = [elem for elem in elements if elem.text]
    if text_elements:
        text_lengths = [len(elem.text) for elem in text_elements]
        avg_length = sum(text_lengths) / len(text_lengths)
        print(f"\n📝 Text Statistics:")
        print(f"  Elements with text: {len(text_elements)}/{len(elements)}")
        print(f"  Average text length: {avg_length:.1f} characters")
        print(f"  Total characters: {sum(text_lengths):,}")


async def show_results_summary(config_name, root_section, tree_report):
    """Show summary of tree building results"""
    
    print(f"\n📈 Results Summary:")
    quality = tree_report.get('quality_metrics', {})
    
    print(f"  🌳 Total sections: {tree_report['total_sections']}")
    print(f"  📏 Max depth: {tree_report['max_depth']}")
    print(f"  📝 Avg content length: {tree_report['avg_content_length']:.1f} chars")
    
    if quality:
        print(f"  ✨ Title quality: {quality.get('title_quality', 0):.1%}")
        print(f"  📄 Content coverage: {quality.get('content_coverage', 0):.1%}")
    
    # Show level distribution
    level_dist = tree_report['level_distribution']
    print(f"  📊 Level distribution: {dict(sorted(level_dist.items()))}")


def pick_best_configuration(results) -> str:
    """Pick the best configuration based on quality metrics"""
    
    scores = {}
    
    for config_name, result in results.items():
        tree_report = result['tree_report']
        quality = tree_report.get('quality_metrics', {})
        
        score = 0
        
        # Quality scoring
        if quality:
            score += quality.get('title_quality', 0) * 30
            score += quality.get('content_coverage', 0) * 20
        
        # Structure scoring
        section_count = tree_report['total_sections']
        if 5 <= section_count <= 25:  # Reasonable section count
            score += 20
        elif section_count > 0:
            score += 10
        
        depth = tree_report['max_depth']
        if 2 <= depth <= 5:  # Good hierarchy depth
            score += 15
        elif depth > 0:
            score += 5
        
        # Content distribution
        if tree_report['leaf_sections'] > 0:
            score += 10
        
        scores[config_name] = score
    
    best_config = max(scores.keys(), key=lambda k: scores[k])
    
    print(f"🏆 Best Configuration Scores:")
    for config, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {config}: {score:.1f}")
    
    return best_config


async def compare_configurations(results):
    """Compare different configurations side by side"""
    
    # Create comparison table
    metrics = [
        ("Total Sections", lambda r: r['tree_report']['total_sections']),
        ("Max Depth", lambda r: r['tree_report']['max_depth']),
        ("Leaf Sections", lambda r: r['tree_report']['leaf_sections']),
        ("Title Quality", lambda r: r['tree_report'].get('quality_metrics', {}).get('title_quality', 0)),
        ("Content Coverage", lambda r: r['tree_report'].get('quality_metrics', {}).get('content_coverage', 0)),
    ]
    
    # Header
    config_names = list(results.keys())
    print(f"{'Metric':<20}", end="")
    for name in config_names:
        short_name = name.split()[0]  # Get emoji + first word
        print(f"{short_name:<15}", end="")
    print()
    print("─" * (20 + 15 * len(config_names)))
    
    # Data rows
    for metric_name, metric_func in metrics:
        print(f"{metric_name:<20}", end="")
        for config_name in config_names:
            value = metric_func(results[config_name])
            if isinstance(value, float):
                if 0 <= value <= 1:  # Probably a percentage
                    print(f"{value:.1%}          ", end="")
                else:
                    print(f"{value:.1f}          ", end="")
            else:
                print(f"{value}             ", end="")
        print()


async def detailed_tree_analysis(result):
    """Perform detailed analysis of the best tree"""
    
    root_section = result['root_section']
    merger = result['merger']
    
    print("🔍 Detailed Tree Structure:")
    merger.tree_builder.print_tree(root_section)
    
    # Analyze tree quality
    print(f"\n🎯 Quality Analysis:")
    
    all_sections = [root_section] + merger.tree_builder._get_all_descendants(root_section)
    content_sections = [s for s in all_sections if s.level >= 0]
    
    if content_sections:
        # Hierarchy analysis
        levels = [s.level for s in content_sections]
        print(f"  📊 Hierarchy levels: {min(levels)} to {max(levels)}")
        
        # Content distribution
        content_lengths = [len(s.content) for s in content_sections if s.content]
        if content_lengths:
            print(f"  📝 Content distribution:")
            print(f"    Min: {min(content_lengths)} chars")
            print(f"    Max: {max(content_lengths)} chars")
            print(f"    Avg: {sum(content_lengths)/len(content_lengths):.1f} chars")
        
        # Show sample sections
        print(f"\n📋 Sample Sections:")
        for i, section in enumerate(content_sections[:5]):
            title = section.title[:30] + "..." if len(section.title) > 30 else section.title
            content = section.content[:50] + "..." if len(section.content) > 50 else section.content
            print(f"  {i+1}. L{section.level}: {title}")
            print(f"     Content: {content}")
            print(f"     Children: {len(section.sub_sections)}")


async def demonstrate_applications(result):
    """Demonstrate practical applications of the section tree"""
    
    root_section = result['root_section']
    
    print("🛠️ Practical Applications:")
    
    # 1. Table of Contents generation
    print("\n📑 1. Table of Contents Generation:")
    toc = generate_table_of_contents(root_section)
    for line in toc[:10]:  # Show first 10 lines
        print(f"    {line}")
    if len(toc) > 10:
        print(f"    ... and {len(toc) - 10} more entries")
    
    # 2. Section search
    print(f"\n🔍 2. Section Search:")
    search_terms = ["条", "第", "规定", "管理", "实施"]  # Common legal terms
    
    for term in search_terms[:3]:  # Show first 3 searches
        matching_sections = search_sections(root_section, term)
        if matching_sections:
            print(f"    Search '{term}': {len(matching_sections)} matches")
            for section in matching_sections[:2]:  # Show first 2 matches
                title = section.title[:30] + "..." if len(section.title) > 30 else section.title
                print(f"      - L{section.level}: {title}")
    
    # 3. Document outline
    print(f"\n📋 3. Document Outline:")
    outline = generate_outline(root_section, max_depth=3)
    for line in outline[:8]:  # Show first 8 lines
        print(f"    {line}")
    
    # 4. Section statistics
    print(f"\n📊 4. Section Statistics:")
    stats = calculate_section_stats(root_section)
    print(f"    Sections by level: {stats['level_counts']}")
    print(f"    Average words per section: {stats['avg_words']:.1f}")
    print(f"    Sections with subsections: {stats['parent_sections']}")


def generate_table_of_contents(root_section, level=0):
    """Generate table of contents from section tree"""
    
    toc = []
    
    for section in root_section.sub_sections:
        if section.title and section.title != "Untitled Section":
            indent = "  " * level
            title = section.title[:60] + "..." if len(section.title) > 60 else section.title
            toc.append(f"{indent}{title}")
            
            # Recursively add subsections
            toc.extend(generate_table_of_contents(section, level + 1))
    
    return toc


def search_sections(root_section, search_term):
    """Search for sections containing a term"""
    
    matches = []
    
    def search_recursive(section):
        # Check title and content
        if (search_term.lower() in (section.title or "").lower() or
            search_term.lower() in (section.content or "").lower()):
            matches.append(section)
        
        # Search children
        for child in section.sub_sections:
            search_recursive(child)
    
    search_recursive(root_section)
    return matches


def generate_outline(root_section, max_depth=None, current_depth=0):
    """Generate document outline"""
    
    outline = []
    
    if max_depth is not None and current_depth >= max_depth:
        return outline
    
    for i, section in enumerate(root_section.sub_sections):
        if section.title and section.title != "Untitled Section":
            prefix = "  " * current_depth + f"{i+1}. "
            title = section.title[:50] + "..." if len(section.title) > 50 else section.title
            outline.append(f"{prefix}{title}")
            
            # Add subsections
            outline.extend(generate_outline(section, max_depth, current_depth + 1))
    
    return outline


def calculate_section_stats(root_section):
    """Calculate various statistics about the section tree"""
    
    def collect_all_sections(section):
        sections = [section]
        for child in section.sub_sections:
            sections.extend(collect_all_sections(child))
        return sections
    
    all_sections = collect_all_sections(root_section)
    content_sections = [s for s in all_sections if s.level >= 0]
    
    # Level counts
    level_counts = {}
    for section in content_sections:
        level_counts[section.level] = level_counts.get(section.level, 0) + 1
    
    # Word counts
    word_counts = []
    for section in content_sections:
        if section.content:
            words = len(section.content.split())
            word_counts.append(words)
    
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    
    # Parent sections
    parent_sections = len([s for s in content_sections if s.sub_sections])
    
    return {
        'level_counts': level_counts,
        'avg_words': avg_words,
        'parent_sections': parent_sections
    }


# python -m models.layout_structuring.content_merger.demo_complete_pipeline
if __name__ == "__main__":
    asyncio.run(main()) 