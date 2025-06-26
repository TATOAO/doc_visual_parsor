#
from doc_chunking.utils.llm import get_llm_client
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult, LayoutElement, ElementType
from doc_chunking.schemas.schemas import Section
from typing import List, Optional, Dict, Any
from .element_encoder import ElementEncoder, FeatureWeights, ElementCode

from doc_chunking.layout_structuring.title_structure_builder_llm.structurer_llm import title_structure_builder_llm





# python -m models.layout_structuring.content_merger.__init__
if __name__ == "__main__":
    import json
    async def main():
        
        # Load JSON data
        with open("./hybrid_extraction_result.json", "r") as f:
            layout_data = json.load(f)
        
        # Create LayoutExtractionResult object
        layout_extraction_result = LayoutExtractionResult(
            elements=layout_data["elements"], 
            metadata=layout_data["metadata"]
        )
        
        # Construct the document sections
        section = await content_merger.construct_section(layout_extraction_result)

        from doc_chunking.naive_llm.helpers import remove_circular_references
        remove_circular_references(section)

        with open("./section_post_merger.json", "w") as f:
            json.dump(section.model_dump(), f, indent=4, ensure_ascii=False)
        