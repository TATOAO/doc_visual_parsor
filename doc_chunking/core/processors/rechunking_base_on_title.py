from processor_pipeline import AsyncProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.layout_structuring.title_structure_builder_llm.section_reconstructor import section_reconstructor
from doc_chunking.schemas import Section
from doc_chunking.utils.logging_config import get_logger
from typing import AsyncGenerator, Any

logger = get_logger(__name__)

class RechunkingBaseOnTitleProcessor(AsyncProcessor):
    meta = {
        "name": "RechunkingBaseOnTitleProcessor",
        "input_type": str,
        "output_type": Any,
    }

    async def process(self, chunk_generator: AsyncGenerator[str, None]) -> AsyncGenerator[Any, None]:

        title_structure = ""
        async for chunk in chunk_generator:
            title_structure =  chunk
        
        if 'layout_extraction_result' in self.session:
            sections = section_reconstructor(title_structure, self.session['layout_extraction_result'])
            yield sections
        else:
            logger.error("layout_extraction_result not found in session, skipping rechunking")
            yield "layout_extraction_result not found in session, skipping rechunking"

        
# python -m doc_chunking.core.processors.rechunking_base_on_title
if __name__ == "__main__":
    import logging
    from processor_pipeline import AsyncPipeline
    from doc_chunking.core.processors.page_chunker import PdfPageImageSplitterProcessor
    from doc_chunking.core.processors.page_image_layout_processor import PageImageLayoutProcessor
    from doc_chunking.core.processors.bbox_nlp_processor import BboxNLPProcessor
    from doc_chunking.core.processors.title_structure_processor import TitleStructureProcessor

    async def main():
        logging.getLogger().setLevel(logging.DEBUG)
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor(), 
            BboxNLPProcessor(),
            TitleStructureProcessor(),
            RechunkingBaseOnTitleProcessor(),
        ])

        async for section in pipeline.astream(input_data='/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'):
            print(section)
    
    import asyncio
    asyncio.run(main())