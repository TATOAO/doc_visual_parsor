import asyncio 
import os
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any
from doc_chunking.schemas.schemas import Section, InputDataType
from doc_chunking.layout_detection.layout_extraction.docx_layout_extractor import DocxLayoutExtrator
from doc_chunking.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor

from doc_chunking.layout_structuring.title_structure_builder_llm.structurer_llm import stream_title_structure_builder_llm
from doc_chunking.layout_structuring.title_structure_builder_llm.section_reconstructor import streaming_section_reconstructor
from doc_chunking.layout_structuring.title_structure_builder_llm.flatten_sections_generator import streaming_flatten_sections_generator, flatten_sections_generator

import logging
logger = logging.getLogger(__name__)

class Chunker:

    def __init__(self):
        # Initialize extractors lazily to avoid loading models at startup
        self._pdf_extractor = None
        self._docx_extractor = None

    def _get_pdf_extractor(self):
        """Lazy initialization of PDF extractor."""
        if self._pdf_extractor is None:
            self._pdf_extractor = PdfStyleCVMixLayoutExtractor()
        return self._pdf_extractor
    
    def _get_docx_extractor(self):
        """Lazy initialization of DOCX extractor."""
        if self._docx_extractor is None:
            self._docx_extractor = DocxLayoutExtrator()
        return self._docx_extractor

    def _get_input_type(self, input_data: InputDataType) -> str:
        """Determine input type based on file extension or content."""
        if isinstance(input_data, (str, Path)):
            file_path = str(input_data)
            _, ext = os.path.splitext(file_path.lower())
            if ext == '.pdf':
                return 'pdf'
            elif ext in ['.docx', '.doc']:
                return 'docx'
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            # For bytes or other types, we would need additional logic
            # For now, assume it's handled by the extractors
            raise ValueError("Cannot determine file type from input data")

    async def chunk_async(self, input_data: InputDataType) -> AsyncGenerator[Section, None]:
        """
        Asynchronously chunk a document into sections.
        
        Args:
            input_data: Path to PDF or DOCX file, or file content
            
        Yields:
            Section objects representing document structure
        """
        try:
            # Determine input type
            input_type = self._get_input_type(input_data)
            
            # Extract layout based on input type
            if input_type == 'pdf':
                logger.info(f"Processing PDF file: {input_data}")
                extractor = self._get_pdf_extractor()
                layout_result = extractor._detect_layout(input_data)
            elif input_type == 'docx':
                logger.info(f"Processing DOCX file: {input_data}")
                extractor = self._get_docx_extractor()
                layout_result = extractor._detect_layout(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            logger.info(f"Extracted {len(layout_result.elements)} layout elements")
            
            # Build title structure using streaming LLM and section reconstructor
            logger.info("Building title structure and reconstructing sections...")
            
            # Use streaming_section_reconstructor to process LLM output and yield Section objects
            async for section in streaming_section_reconstructor(
                stream_title_structure_builder_llm(layout_result), 
                layout_result
            ):
                yield section

            logger.info("Chunking completed")
            
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            # Yield an error section
            error_section = Section(
                title="Error",
                content=f"Failed to process document: {str(e)}",
                level=0
            )
            yield error_section

    async def chunk_flat_async(self, input_data: InputDataType) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Asynchronously chunk a document into flattened sections.
        
        Args:
            input_data: Path to PDF or DOCX file, or file content
            
        Yields:
            Lists of dictionaries with flattened section data containing:
            - 'title': The title joined by all parent titles with format "titleA-titleA1-..."
            - 'content': The content of the section
            - 'level': The level of the section
        """
        try:
            # Determine input type
            input_type = self._get_input_type(input_data)
            
            # Extract layout based on input type
            if input_type == 'pdf':
                logger.info(f"Processing PDF file for flattening: {input_data}")
                extractor = self._get_pdf_extractor()
                layout_result = extractor._detect_layout(input_data)
            elif input_type == 'docx':
                logger.info(f"Processing DOCX file for flattening: {input_data}")
                extractor = self._get_docx_extractor()
                layout_result = extractor._detect_layout(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            logger.info(f"Extracted {len(layout_result.elements)} layout elements")
            
            # Build title structure using streaming LLM and flatten sections generator
            logger.info("Building title structure and flattening sections...")
            
            # Use streaming_flatten_sections_generator to process LLM output and yield flattened sections
            async for flattened_sections in streaming_flatten_sections_generator(
                stream_title_structure_builder_llm(layout_result), 
                layout_result
            ):
                yield flattened_sections

            logger.info("Flattening completed")
            
        except Exception as e:
            logger.error(f"Error during flattening: {str(e)}")
            # Yield an error section
            error_section = [{
                'title': "Error",
                'content': f"Failed to process document: {str(e)}",
                'level': 0
            }]
            yield error_section

    def chunk(self, input_data: InputDataType) -> Section:
        """
        Synchronously chunk a document into sections.
        
        Args:
            input_data: Path to PDF or DOCX file, or file content
            
        Returns:
            Root Section object representing document structure
        """
        async def _collect_sections():
            sections = []
            async for section in self.chunk_async(input_data):
                sections.append(section)
                logger.info(f"Collected section: {len(section.sub_sections)}")
            
            if not sections:
                return Section(title="Empty Document", content="", level=0)
            elif len(sections) == 1:
                return sections[0]
            else:
                # If multiple sections, create a root section containing them
                root = Section(
                    title="Document",
                    content="",
                    level=0,
                    sub_sections=sections
                )
                return root
        
        return asyncio.run(_collect_sections())

    def chunk_flat(self, input_data: InputDataType) -> List[Dict[str, Any]]:
        """
        Synchronously chunk a document into flattened sections.
        
        Args:
            input_data: Path to PDF or DOCX file, or file content
            
        Returns:
            List of dictionaries with flattened section data containing:
            - 'title': The title joined by all parent titles with format "titleA-titleA1-..."
            - 'content': The content of the section
            - 'level': The level of the section
        """
        async def _collect_flat_sections():
            all_sections = []
            async for flattened_sections in self.chunk_flat_async(input_data):
                # Keep only the latest/most complete version
                all_sections = flattened_sections
                logger.info(f"Collected {len(flattened_sections)} flattened sections")
            
            return all_sections or []
        
        return asyncio.run(_collect_flat_sections())

# python -m models.documents_chunking.chunker
if __name__ == "__main__":


    def test_chunker():
        chunker = Chunker()
        word_path = "tests/test_data/1-1 买卖合同（通用版）.docx"
        section_result = chunker.chunk(word_path)
        print(f"Result: {section_result.title}")
        print(f"Content length: {len(section_result.content)}")

        from doc_chunking.utils.helper import remove_circular_references
        remove_circular_references(section_result)
        import json
        json.dump(section_result.model_dump(), open("section_result.json", "w"), indent=2, ensure_ascii=False)

    async def test_chunker_async():
        chunker = Chunker()
        word_path = "tests/test_data/1-1 买卖合同（通用版）.docx"
        async for section in chunker.chunk_async(word_path):
            print(f"Result: {section}")

    asyncio.run(test_chunker_async())
