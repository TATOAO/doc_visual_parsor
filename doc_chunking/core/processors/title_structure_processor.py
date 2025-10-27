import asyncio
from typing import Any, List, AsyncGenerator, Tuple
from processor_pipeline import AsyncProcessor
from .page_chunker import PdfPageImageSplitterProcessor
from .page_image_layout_processor import PageImageLayoutProcessor
from .bbox_nlp_processor import BboxNLPProcessor
from doc_chunking.schemas import LayoutElement, LayoutExtractionResult
from doc_chunking.layout_structuring.title_structure_builder_llm.structurer_llm import stream_title_structure_builder_llm_with_plain_text

from doc_chunking.utils.logging_config import get_logger

logger = get_logger(__name__)

class TitleStructureProcessor(AsyncProcessor):
    meta = {
        "name": "TitleStructureProcessor",
        "input_type": Tuple[str, LayoutElement],
        "output_type": str,
    }

    async def process(self, input_data: AsyncGenerator[Tuple[str, LayoutElement], None]) -> AsyncGenerator[str, None]:

        full_text = ""
        elements_result = []
        async for line, element in input_data:
            full_text += line
            logger.info(f'collecting elements: {element}')
            elements_result.append(element)
        
        layout_extraction_result = LayoutExtractionResult(
            elements=elements_result,
            metadata=None
        )
        self.session['layout_extraction_result'] = layout_extraction_result


        logger.info("Collection done")
        logger.debug(f"TitleStructureProcessor processed full text: {full_text}")
        
        next_title_structure = ""
        async for chunk in stream_title_structure_builder_llm_with_plain_text(next_title_structure, full_text):
            next_title_structure += chunk
            logger.debug(f"TitleStructureProcessor processed chunk: {chunk}")
        
        # yield the final title structure
        yield next_title_structure


# python -m doc_chunking.core.processors.title_structure_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    import logging
    input_file_path = '/home/tatoao-ubuntu/Downloads/tdjyqht.pdf'
    async def main():
        logging.getLogger().setLevel(logging.INFO)
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor(), 
            BboxNLPProcessor(),
            TitleStructureProcessor()
            ]
        )
        async for item in pipeline.astream(input_data=input_file_path):
            print(item)
        
        print(pipeline.session)

    import asyncio
    asyncio.run(main())