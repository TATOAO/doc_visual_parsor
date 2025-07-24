import json
from processor_pipeline import AsyncProcessor
from doc_chunking.schemas.layout_schemas import LayoutElement
from doc_chunking.layout_structuring.title_structure_builder_llm.section_reconstructor import section_reconstructor
from doc_chunking.layout_structuring.title_structure_builder_llm.flatten_sections_generator import _flatten_section_tree
from doc_chunking.schemas import Section
from doc_chunking.utils.logging_config import get_logger
from typing import AsyncGenerator, Any

logger = get_logger(__name__)

class RechunkingBaseOnTitleProcessor(AsyncProcessor):
    meta = {
        "name": "RechunkingBaseOnTitleProcessor",
        "input_type": str,
        "output_type": Section,
    }

    async def process(self, chunk_generator: AsyncGenerator[str, None]) -> AsyncGenerator[Section, None]:

        title_structure = ""
        async for chunk in chunk_generator:
            title_structure =  chunk
        
        if 'layout_extraction_result' in self.session:
            sections = section_reconstructor(title_structure, self.session['layout_extraction_result'])

            flattened_sections = _flatten_section_tree(sections)

            for section in flattened_sections:
                yield section
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
    from doc_chunking.utils.logging_config import configure_for_development
    configure_for_development()
    
    async def main():
        pipeline = AsyncPipeline([
            PdfPageImageSplitterProcessor(), 
            PageImageLayoutProcessor(), 
            BboxNLPProcessor(),
            TitleStructureProcessor(),
            RechunkingBaseOnTitleProcessor(),
        ])

        input_data = '/Users/tatoao_mini/Work/Kindee/ai_contract/合同样例/智能设备采购及运维合作合同.docx'
        index = 0

        from doc_chunking.utils.helper import remove_circular_references
        async for section in pipeline.astream(input_data=input_data):
            print('-'*100)
            index += 1
            if section:
                with open(f'result_{index}.json', 'w', encoding='utf-8') as f:
                    remove_circular_references(section)
                    json.dump(section.model_dump(), f, indent=4, ensure_ascii=False)  
        
        print(f"Processing completed. Generated {index} result files.")
    
    import asyncio
    asyncio.run(main())
    import json