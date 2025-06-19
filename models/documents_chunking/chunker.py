import asyncio 
import os
from pathlib import Path
from typing import AsyncGenerator
from models.schemas.schemas import Section, InputDataType
from models.layout_detection.layout_extraction.docx_layout_extractor import DocxLayoutExtrator
from models.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor

from models.layout_structuring.title_structure_builder_llm.structurer_llm import stream_title_structure_builder_llm
from models.layout_structuring.title_structure_builder_llm.section_reconstructor import streaming_section_reconstructor

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
                print(f"Processing PDF file: {input_data}")
                extractor = self._get_pdf_extractor()
                layout_result = extractor._detect_layout(input_data)
            elif input_type == 'docx':
                print(f"Processing DOCX file: {input_data}")
                extractor = self._get_docx_extractor()
                layout_result = extractor._detect_layout(input_data)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
            
            print(f"Extracted {len(layout_result.elements)} layout elements")
            
            # Build title structure using streaming LLM and section reconstructor
            print("Building title structure and reconstructing sections...")
            
            # Use streaming_section_reconstructor to process LLM output and yield Section objects
            async for section in streaming_section_reconstructor(
                stream_title_structure_builder_llm(layout_result), 
                layout_result
            ):
                yield section
            
        except Exception as e:
            print(f"Error during chunking: {str(e)}")
            # Yield an error section
            error_section = Section(
                title="Error",
                content=f"Failed to process document: {str(e)}",
                level=0
            )
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

                print(f"Collected section: {len(section.sub_sections)}")
            
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

# python -m models.documents_chunking.chunker
if __name__ == "__main__":


    def test_chunker():
        chunker = Chunker()
        word_path = "tests/test_data/1-1 买卖合同（通用版）.docx"
        section_result = chunker.chunk(word_path)
        print(f"Result: {section_result.title}")
        print(f"Content length: {len(section_result.content)}")

        from models.naive_llm.helpers.section_token_parsor import remove_circular_references
        remove_circular_references(section_result)
        import json
        json.dump(section_result.model_dump(), open("section_result.json", "w"), indent=2, ensure_ascii=False)

    async def test_chunker_async():
        chunker = Chunker()
        word_path = "tests/test_data/1-1 买卖合同（通用版）.docx"
        async for section in chunker.chunk_async(word_path):
            print(f"Result: {section}")

    asyncio.run(test_chunker_async())
