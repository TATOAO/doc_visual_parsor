"""
Enhanced position mapping that works with different document types
Extends the original tree_like_structure_mapping.py to work with PDF, DOCX, and text documents
"""

from models.utils.schemas import Section, Positions, BoundingBox, DocumentType
from typing import List, Tuple, Union, Dict, Any, Optional
from fuzzysearch import find_near_matches
import fitz  # PyMuPDF
import docx
from pathlib import Path


class DocumentPositionMapper:
    """
    Maps section positions for different document types
    """
    
    def __init__(self, document_path: Union[str, Path], document_type: DocumentType):
        self.document_path = Path(document_path)
        self.document_type = document_type
        self._doc_data = None
        self._raw_text = None
        
    def _load_document_data(self):
        """Load document-specific data structures"""
        if self._doc_data is not None:
            return
            
        if self.document_type == DocumentType.PDF:
            self._load_pdf_data()
        elif self.document_type == DocumentType.DOCX:
            self._load_docx_data()
        elif self.document_type == DocumentType.TEXT:
            self._load_text_data()
    
    def _load_pdf_data(self):
        """Load PDF with text and coordinate information"""
        pdf_doc = fitz.open(str(self.document_path))
        
        self._doc_data = {
            'pages': [],
            'text_blocks': [],
            'raw_text_parts': []
        }
        
        raw_text_parts = []
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            
            # Extract text blocks with coordinates
            text_dict = page.get_text("dict")
            page_blocks = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                bbox = span["bbox"]  # [x0, y0, x1, y1]
                                block_info = {
                                    'text': text,
                                    'bbox': bbox,
                                    'page_number': page_num,
                                    'font_size': span.get('size', 12),
                                    'font_flags': span.get('flags', 0),
                                    'global_start_idx': len(''.join(raw_text_parts)),
                                    'global_end_idx': len(''.join(raw_text_parts)) + len(text)
                                }
                                page_blocks.append(block_info)
                                raw_text_parts.append(text + ' ')
            
            self._doc_data['pages'].append(page_blocks)
            self._doc_data['text_blocks'].extend(page_blocks)
        
        self._raw_text = ''.join(raw_text_parts)
        pdf_doc.close()
    
    def _load_docx_data(self):
        """Load DOCX with paragraph information"""
        doc = docx.Document(str(self.document_path))
        
        self._doc_data = {
            'paragraphs': [],
            'raw_text_parts': []
        }
        
        raw_text_parts = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                para_info = {
                    'text': paragraph.text,
                    'paragraph_index': i,
                    'style_name': paragraph.style.name,
                    'global_start_idx': len('\n\n'.join(raw_text_parts)),
                    'global_end_idx': len('\n\n'.join(raw_text_parts)) + len(paragraph.text)
                }
                self._doc_data['paragraphs'].append(para_info)
                raw_text_parts.append(paragraph.text)
        
        self._raw_text = '\n\n'.join(raw_text_parts)
    
    def _load_text_data(self):
        """Load plain text file"""
        with open(self.document_path, 'r', encoding='utf-8') as f:
            self._raw_text = f.read()
        self._doc_data = {'raw_text': self._raw_text}
    
    def find_ordered_fuzzy_sequence_enhanced(self, keywords: List[str], max_l_dist: int = 3) -> List[Positions]:
        """
        Enhanced version that returns Position objects instead of tuples
        """
        self._load_document_data()
        
        if self.document_type == DocumentType.TEXT:
            return self._find_text_positions(keywords, max_l_dist)
        elif self.document_type == DocumentType.PDF:
            return self._find_pdf_positions(keywords, max_l_dist)
        elif self.document_type == DocumentType.DOCX:
            return self._find_docx_positions(keywords, max_l_dist)
        else:
            raise ValueError(f"Unsupported document type: {self.document_type}")
    
    def _find_text_positions(self, keywords: List[str], max_l_dist: int) -> List[Positions]:
        """Find positions in plain text"""
        positions = []
        cursor = 0
        
        for keyword in keywords:
            sub_text = self._raw_text[cursor:]
            result = find_near_matches(keyword, sub_text, max_l_dist=max_l_dist)
            if not result:
                return []  # Keyword not found
                
            match = result[0]
            start, end = match.start + cursor, match.end + cursor
            
            position = Positions.from_text(start, end, keyword=keyword)
            positions.append(position)
            cursor = end
        
        return positions
    
    def _find_pdf_positions(self, keywords: List[str], max_l_dist: int) -> List[Positions]:
        """Find positions in PDF with coordinate information"""
        positions = []
        cursor = 0
        
        for keyword in keywords:
            sub_text = self._raw_text[cursor:]
            result = find_near_matches(keyword, sub_text, max_l_dist=max_l_dist)
            if not result:
                return []  # Keyword not found
                
            match = result[0]
            global_start = match.start + cursor
            global_end = match.end + cursor
            
            # Find which text block(s) contain this match
            matching_block = None
            for block in self._doc_data['text_blocks']:
                if (block['global_start_idx'] <= global_start <= block['global_end_idx'] or
                    block['global_start_idx'] <= global_end <= block['global_end_idx']):
                    matching_block = block
                    break
            
            if matching_block:
                bbox = BoundingBox(
                    x1=matching_block['bbox'][0],
                    y1=matching_block['bbox'][1],
                    x2=matching_block['bbox'][2],
                    y2=matching_block['bbox'][3],
                    page_number=matching_block['page_number']
                )
                
                position = Positions.from_pdf(
                    page_number=matching_block['page_number'],
                    bounding_box=bbox,
                    text_start=global_start,
                    text_end=global_end,
                    keyword=keyword,
                    font_size=matching_block['font_size']
                )
            else:
                # Fallback to text-only position
                position = Positions.from_text(global_start, global_end, keyword=keyword)
            
            positions.append(position)
            cursor = global_end
        
        return positions
    
    def _find_docx_positions(self, keywords: List[str], max_l_dist: int) -> List[Positions]:
        """Find positions in DOCX with paragraph information"""
        positions = []
        cursor = 0
        
        for keyword in keywords:
            sub_text = self._raw_text[cursor:]
            result = find_near_matches(keyword, sub_text, max_l_dist=max_l_dist)
            if not result:
                return []  # Keyword not found
                
            match = result[0]
            global_start = match.start + cursor
            global_end = match.end + cursor
            
            # Find which paragraph contains this match
            matching_para = None
            for para in self._doc_data['paragraphs']:
                if para['global_start_idx'] <= global_start <= para['global_end_idx']:
                    matching_para = para
                    break
            
            if matching_para:
                # Calculate character position within the paragraph
                char_start = global_start - matching_para['global_start_idx']
                char_end = global_end - matching_para['global_start_idx']
                
                position = Positions.from_docx(
                    paragraph_index=matching_para['paragraph_index'],
                    character_start=char_start,
                    character_end=char_end,
                    keyword=keyword,
                    style=matching_para['style_name']
                )
            else:
                # Fallback to text-only position
                position = Positions.from_text(global_start, global_end, keyword=keyword)
            
            positions.append(position)
            cursor = global_end
        
        return positions


def set_section_position_enhanced(section_tree: Section, document_path: Union[str, Path], 
                                document_type: DocumentType) -> Section:
    """
    Enhanced version of set_section_position_index that works with different document types
    """
    mapper = DocumentPositionMapper(document_path, document_type)
    
    # Flatten the tree to get all sections
    sections = flatten_section_tree_to_tokens(section_tree)
    title_list = [section.title for section in sections]
    
    # Find positions using the enhanced method
    positions = mapper.find_ordered_fuzzy_sequence_enhanced(title_list)
    
    if len(positions) != len(sections):
        print(f"Warning: Found {len(positions)} positions for {len(sections)} sections")
        return section_tree
    
    # Set title positions using the new Position objects
    for section, position in zip(sections, positions):
        section.title_position = position
        
        # Also set legacy fields for backward compatibility
        if position.text_position:
            section.title_position_index = (position.text_position.start, position.text_position.end)
        elif position.pdf_position and position.pdf_position.text_start is not None:
            section.title_position_index = (position.pdf_position.text_start, position.pdf_position.text_end)
    
    # Set content positions (similar logic as original but with Position objects)
    _set_content_positions_enhanced(section_tree, sections, mapper)
    
    return section_tree


def _set_content_positions_enhanced(section_tree: Section, flattened_sections: List[Section], 
                                  mapper: DocumentPositionMapper):
    """Set content positions using enhanced position system"""
    
    def set_content_positions_recursive(section: Section):
        """Recursively set content positions"""
        
        if section.sub_sections:
            # Process subsections first
            for sub_section in section.sub_sections:
                set_content_positions_recursive(sub_section)
            
            # Content spans from title end to last subsection's content end
            last_subsection = section.sub_sections[-1]
            if section.title_position and last_subsection.content_position:
                # Create content position based on document type
                if mapper.document_type == DocumentType.PDF and section.title_position.pdf_position:
                    section.content_position = Positions.from_pdf(
                        page_number=section.title_position.pdf_position.page_number,
                        text_start=section.title_position.pdf_position.text_end,
                        text_end=last_subsection.content_position.pdf_position.text_end if last_subsection.content_position.pdf_position else None
                    )
                elif mapper.document_type == DocumentType.DOCX and section.title_position.docx_position:
                    section.content_position = Positions.from_docx(
                        paragraph_index=section.title_position.docx_position.paragraph_index,
                        character_start=section.title_position.docx_position.character_end,
                        character_end=last_subsection.content_position.docx_position.character_end if last_subsection.content_position.docx_position else None
                    )
                else:
                    # Fallback to text position
                    start_pos = section.title_position.text_position.end if section.title_position.text_position else 0
                    end_pos = last_subsection.content_position.text_position.end if last_subsection.content_position and last_subsection.content_position.text_position else len(mapper._raw_text)
                    section.content_position = Positions.from_text(start_pos, end_pos)
        else:
            # Leaf section - content extends to next section or end
            current_idx = flattened_sections.index(section)
            
            if current_idx < len(flattened_sections) - 1:
                next_section = flattened_sections[current_idx + 1]
                if section.title_position and next_section.title_position:
                    # Content extends to start of next section
                    if mapper.document_type == DocumentType.PDF and section.title_position.pdf_position:
                        section.content_position = Positions.from_pdf(
                            page_number=section.title_position.pdf_position.page_number,
                            text_start=section.title_position.pdf_position.text_end,
                            text_end=next_section.title_position.pdf_position.text_start if next_section.title_position.pdf_position else None
                        )
                    elif mapper.document_type == DocumentType.DOCX and section.title_position.docx_position:
                        section.content_position = Positions.from_docx(
                            paragraph_index=section.title_position.docx_position.paragraph_index,
                            character_start=section.title_position.docx_position.character_end,
                            character_end=next_section.title_position.docx_position.character_start if next_section.title_position.docx_position else None
                        )
                    else:
                        start_pos = section.title_position.text_position.end if section.title_position.text_position else 0
                        end_pos = next_section.title_position.text_position.start if next_section.title_position and next_section.title_position.text_position else len(mapper._raw_text)
                        section.content_position = Positions.from_text(start_pos, end_pos)
            else:
                # Last section - content extends to end of document
                if section.title_position:
                    if mapper.document_type == DocumentType.PDF and section.title_position.pdf_position:
                        section.content_position = Positions.from_pdf(
                            page_number=section.title_position.pdf_position.page_number,
                            text_start=section.title_position.pdf_position.text_end,
                            text_end=len(mapper._raw_text)
                        )
                    elif mapper.document_type == DocumentType.DOCX and section.title_position.docx_position:
                        section.content_position = Positions.from_docx(
                            paragraph_index=section.title_position.docx_position.paragraph_index,
                            character_start=section.title_position.docx_position.character_end,
                            character_end=len(mapper._raw_text)
                        )
                    else:
                        start = section.title_position.text_position.end if section.title_position.text_position else 0
                        section.content_position = Positions.from_text(start, len(mapper._raw_text))
    
    set_content_positions_recursive(section_tree)


def flatten_section_tree_to_tokens(section_tree: Section) -> List[Section]:
    """
    Flatten the section tree into a flat list of Section objects
    (Copied from original for convenience)
    """
    def _flatten_section(section: Section) -> List[Section]:
        sections = [section]
        for sub_section in section.sub_sections:
            sections.extend(_flatten_section(sub_section))
        return sections
    
    if section_tree.sub_sections:
        all_sections = [section_tree]
        for section in section_tree.sub_sections:
            all_sections.extend(_flatten_section(section))
        return all_sections
    else:
        return _flatten_section(section_tree)


# Example usage
# python -m models.naive_llm.helpers.enhanced_position_mapping
if __name__ == "__main__":
    import json
    
    # Test with DOCX
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    # Test with pdf
    # docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.pdf"
    
    # Load section tree
    with open('section_tree.json', 'r') as f:
        j = json.load(f)
    section_tree = Section.model_validate(j)
    
    # Enhanced position mapping
    enhanced_tree = set_section_position_enhanced(
        section_tree, 
        docx_path, 
        DocumentType.TEXT
    )

    with open('enhanced_section_tree_text.json', 'w') as f:
        json.dump(enhanced_tree.model_dump(), f, indent=4, ensure_ascii=False)
    
    # # Print results
    # for section in flatten_section_tree_to_tokens(enhanced_tree):
    #     print(f"Level {section.level}: {section.title}")
    #     if section.title_position:
    #         print(f"  Title position: {section.title_position}")
    #     if section.content_position:
    #         print(f"  Content position: {section.content_position}")
    #     print() 