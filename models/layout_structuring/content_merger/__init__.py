
#
from models.utils.llm import get_llm_client
from models.schemas.layout_schemas import LayoutExtractionResult, LayoutElement, ElementType
from models.schemas.schemas import Section
from typing import List
from .helper import (
    TitleNumberType, 
    TitleNumberUnit, 
    TitleNumber, 
    compare_title_numbers, 
    get_title_hierarchy_path, 
    title_number_extraction
)

class ContentMerger:
    """
    This class is used to merge the content of the elements.
    
    """
    def __init__(self):
        self.llm = get_llm_client()


    async def construct_section(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """

        The general idea is to recursively iterate through the elements
        1. for each element and next element, 
            a. if sibling, we need to merge the content
        2. we need to return the section

        until no more merge can be done

        """


        async for element, next_element in zip(elements, elements[1:]):
            is_sibling = await self.is_sibling(element, next_element)
            if is_sibling:
                content = await self.merge_content(element, next_element)
                element.text = content
            else:
                await self.construct_section(element)
        
        return elements

        
    

    async def is_sibling(self, element1: LayoutElement, element2: LayoutElement) -> bool:
        """
        Check if two elements are siblings
        """
        pass



# python -m models.layout_structuring.__init__
if __name__ == "__main__":
    async def main():
        content_merger = ContentMerger()
        import json
        with open("hybrid_extraction_result.json", "r") as f:
            layout_extraction_result = json.load(f)
        layout_extraction_result = LayoutExtractionResult(elements=layout_extraction_result["elements"], 
                                                          metadata=layout_extraction_result["metadata"])
        section = await content_merger.construct_section(layout_extraction_result)
        print(section)

    import asyncio
    asyncio.run(main())


# python -m models.layout_structuring.content_merger.__init__
if __name__ == "__main__":
    async def main():
        content_merger = ContentMerger()
        import json
        
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

        from models.naive_llm.helpers import remove_circular_references
        remove_circular_references(section)

        with open("./section_post_merger.json", "w") as f:
            json.dump(section.model_dump(), f, indent=4, ensure_ascii=False)
        