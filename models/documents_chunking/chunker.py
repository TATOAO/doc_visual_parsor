import asyncio 
from typing import AsyncGenerator
from models.schemas.schemas import Sector, InputDataType
from models.layout_detection.layout_extraction.docx_layout_extractor import DocxLayoutExtrator
from models.layout_detection.layout_extraction.pdf_style_cv_mix_extractor import PdfStyleCVMixLayoutExtractor

from models.layout_structuring.title_structure_builder_llm.structurer_llm import title_structure_builder_llm

class Chunker:

    def __init__(self):
        # self.cv_detector = self._initialize_cv_detector()
        pass

    async def chunk_async(self, input_data: InputDataType) -> AsyncGenerator[Sector, None]:
        pass

    def chunk(self, input_data: InputDataType) -> Sector:
        return asyncio.run(self.chunk_async(input_data))


if __name__ == "__main__":
    chunker = Chunker()
    word_path = "tests/test_data/1-1 买卖合同（通用版）.docx"
    sector_result = chunker.chunk(word_path)

