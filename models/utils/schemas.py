from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Self

class Section(BaseModel):
    title: str = Field(description="The title of the section", default="")
    content: str = Field(description="The content of the section", default="")
    level: int = Field(description="The level of the section", default=0)
    paragraph_index: int = Field(description="The paragraph index of the section", default=0)
    style: str = Field(description="The style of the section", default="")
    confidence: float = Field(description="The confidence of the section", default=0.0)
    bounding_box: List[int] = Field(description="The bounding box of the section", default=[])
    page_number: int = Field(description="The page number of the section", default=0)
    page_index: int = Field(description="The page index of the section", default=0)

    sub_sections: List[Self] = Field(description="The sub sections of the section", default=[])
    parent_section: Optional[Self] = Field(description="The parent section of the section", default=None)


def get_section_tree(section: Self) -> Self:
    """
    Get the section tree of the section
    """
    return section
