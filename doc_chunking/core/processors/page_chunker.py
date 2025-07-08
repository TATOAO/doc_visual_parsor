from processor_pipeline import AsyncProcessor
from typing import Any, AsyncGenerator, Union
from pathlib import Path
from doc_chunking.utils.word_convert_pdf import convert_docx_to_pdf
from doc_chunking.schemas.files import FileInputData

class PageChunker(AsyncProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process(self, input_data: FileInputData) -> AsyncGenerator[Any, None]:
        """
        Chunk a page into sections.

        Args:
            input_data: The input data to process.

        Returns:
            An async generator that yields the chunked sections.
        """

        # if docx convert docx into pdf
        if isinstance(input_data, Path) and input_data.suffix.lower() == '.docx':
            input_data = convert_docx_to_pdf(input_data)





        return input_data