"""
Example: Using doc_chunking library components directly.

This example demonstrates how to use the doc_chunking library
components directly in your Python code without the FastAPI server.
"""

import asyncio
from pathlib import Path
from doc_chunking import Chunker, extract_pdf_pages_into_images, extract_docx_content

async def example_document_processing():
    """Example of processing documents using the library directly."""
    
    # Initialize the chunker
    chunker = Chunker()
    
    # Example 1: Process a PDF file (if available)
    test_pdf = Path("tests/test_data/sample.pdf")
    if test_pdf.exists():
        print(f"📄 Processing PDF: {test_pdf}")
        
        # Chunk the document
        try:
            sections = []
            async for section in chunker.chunk_async(str(test_pdf)):
                sections.append(section)
                print(f"  📝 Section: {section.title}")
                print(f"     Content preview: {section.content[:100]}...")
            
            print(f"  ✅ Extracted {len(sections)} sections")
        except Exception as e:
            print(f"  ❌ Error processing PDF: {e}")
    
    # Example 2: Process a DOCX file (if available)
    test_docx = Path("tests/test_data/1-1 买卖合同（通用版）.docx")
    if test_docx.exists():
        print(f"\n📄 Processing DOCX: {test_docx}")
        
        try:
            sections = []
            async for section in chunker.chunk_async(str(test_docx)):
                sections.append(section)
                print(f"  📝 Section: {section.title}")
                print(f"     Content preview: {section.content[:100]}...")
            
            print(f"  ✅ Extracted {len(sections)} sections")
        except Exception as e:
            print(f"  ❌ Error processing DOCX: {e}")
    
    # Example 3: Extract PDF pages as images
    if test_pdf.exists():
        print(f"\n🖼️  Extracting PDF pages as images...")
        try:
            with open(test_pdf, 'rb') as f:
                images = extract_pdf_pages_into_images(f)
                print(f"  ✅ Extracted {len(images)} page images")
                for i, img in enumerate(images):
                    print(f"    Page {i+1}: {img.size[0]}x{img.size[1]} pixels")
        except Exception as e:
            print(f"  ❌ Error extracting images: {e}")
    
    # Example 4: Extract DOCX content
    if test_docx.exists():
        print(f"\n📝 Extracting DOCX content...")
        try:
            with open(test_docx, 'rb') as f:
                content = extract_docx_content(f)
                print(f"  ✅ Extracted {len(content)} characters")
                print(f"  Content preview: {content[:200]}...")
        except Exception as e:
            print(f"  ❌ Error extracting content: {e}")

def example_layout_detection():
    """Example of using layout detection components."""
    
    from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor
    
    print("\n🔍 Layout Detection Example")
    
    # Initialize the detector
    detector = PdfStyleCVMixLayoutExtractor(
        cv_model_name="docstructbench",
        cv_confidence_threshold=0.25,
        cv_image_size=1024
    )
    
    print(f"  📊 Detector initialized with model: docstructbench")
    print(f"  🎯 Confidence threshold: 0.25")
    print(f"  📐 Image size: 1024")
    
    # Note: In a real scenario, you would call detector.detect(image_path)
    # and detector.visualize(image_path, detection_result)

def example_schema_usage():
    """Example of using the schema classes directly."""
    
    from doc_chunking.schemas.schemas import Section
    
    print("\n📋 Schema Usage Example")
    
    # Create a section manually
    section = Section(
        title="Introduction",
        content="This is the introduction section of the document.",
        level=0,
        element_id=1
    )
    
    # Create a subsection
    subsection = Section(
        title="Background",
        content="This section provides background information.",
        level=1,
        element_id=2,
        parent_section=section
    )
    
    # Add subsection to main section
    section.sub_sections = [subsection]
    
    print(f"  📝 Created section: {section.title}")
    print(f"  📝 With subsection: {subsection.title}")
    print(f"  📊 Total content length: {len(section.content) + len(subsection.content)}")

if __name__ == "__main__":
    print("🚀 Doc Chunking Library Usage Examples")
    print("=" * 50)
    
    # Run async examples
    asyncio.run(example_document_processing())
    
    # Run sync examples
    example_layout_detection()
    example_schema_usage()
    
    print("\n✅ All examples completed!")
    print("\nTo run these examples with real documents:")
    print("1. Place test files in the tests/test_data/ directory")
    print("2. Run: python examples/library_usage_example.py") 