import asyncio
from typing import Any, List, AsyncGenerator, Tuple
from processor_pipeline import AsyncProcessor
from .page_chunker import PdfPageImageSplitterProcessor
from .page_image_layout_processor import PageImageLayoutProcessor
from .bbox_nlp_processor import BboxNLPProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.layout_structuring.title_structure_builder_llm.structurer_llm import stream_title_structure_builder_llm_with_plain_text


class TitleStructureProcessor(AsyncProcessor):
    meta = {
        "name": "TitleStructureProcessor",
        "input_type": Tuple[str, LayoutElement],
        "output_type": Any,
    }

    async def process(self, input_data: AsyncGenerator[Tuple[str, LayoutElement], None]) -> AsyncGenerator[Any, None]:
        line_window = []
        line_window_size = 30
        title_structure = ""
        async for line, element in input_data:
            line_window.append(line)

            # Process when we reach the window size
            next_title_structure = ""
            if len(line_window) == line_window_size:
                async for chunk in stream_title_structure_builder_llm_with_plain_text(title_structure, "\n".join(line_window)):
                    next_title_structure += chunk
                    print(chunk)
                # Clear the window after processing
                line_window = []
            title_structure = next_title_structure

            await asyncio.sleep(0.001)
        
        # Process any remaining lines in the window
        final_result = ""
        if line_window:
            async for chunk in stream_title_structure_builder_llm_with_plain_text(title_structure, "\n".join(line_window)):
                final_result += chunk
                print(chunk)

        yield final_result

        

# python -m doc_chunking.core.processors.title_structure_processor
if __name__ == "__main__":
    from processor_pipeline import AsyncPipeline
    async def main():
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor(), 
            BboxNLPProcessor(),
            TitleStructureProcessor()
            ]
        )
        async for item in pipeline.astream(input_data='/Users/tatoaoliang/Downloads/Work/doc_chunking/tests/test_data/1-1 买卖合同（通用版）.pdf'):
            print(item)

    import asyncio
    asyncio.run(main())