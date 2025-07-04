#!/usr/bin/env python3
"""
Test script to measure timing of chunk_document_sse function,
specifically focusing on CV model loading time.
"""

import asyncio
import time
import logging
from pathlib import Path
from doc_chunking.documents_chunking.chunker import Chunker

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def test_chunk_document_sse_timing():
    """Test the timing of chunk_document_sse with detailed logging."""
    
    # Test with both PDF and DOCX files
    test_files = [
        "tests/test_data/1-1 买卖合同（通用版）.pdf",
        "tests/test_data/1-1 买卖合同（通用版）.docx"
    ]
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"Test file not found: {test_file}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing file: {test_file}")
        print(f"{'='*60}")
        
        # Measure total time
        total_start = time.time()
        
        # Create chunker (this is where lazy initialization happens)
        print("Creating chunker...")
        chunker_start = time.time()
        chunker = Chunker()
        chunker_time = time.time() - chunker_start
        print(f"Chunker created in {chunker_time:.3f} seconds")
        
        # Process the document
        print("Processing document...")
        processing_start = time.time()
        
        sections = []
        async for section in chunker.chunk_async(test_file):
            sections.append(section)
            print(f"  - Received section: {section.title[:50]}...")
        
        processing_time = time.time() - processing_start
        total_time = time.time() - total_start
        
        print(f"\nTiming Summary:")
        print(f"  Chunker creation: {chunker_time:.3f} seconds")
        print(f"  Document processing: {processing_time:.3f} seconds")
        print(f"  Total time: {total_time:.3f} seconds")
        print(f"  Sections processed: {len(sections)}")
        print(f"  Average time per section: {processing_time/len(sections):.3f} seconds" if sections else "No sections")

if __name__ == "__main__":
    asyncio.run(test_chunk_document_sse_timing()) 