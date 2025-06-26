from doc_chunking.naive_llm.agents.control import cut_deeper_node
from doc_chunking.naive_llm.helpers import flatten_section_tree_to_tokens, remove_circular_references
from doc_chunking.schemas.schemas import Section
import asyncio

# python -m models.naive_llm.tests.test_deeper_cuts
if __name__ == "__main__":

    # test deeper 
    from backend.docx_processor import extract_docx_content
    import json

    with open('section_tree_cut.json', 'r') as f:
        section_tree = Section.model_validate_json(f.read())


    section_tree_no_sub_sections = [section for section in flatten_section_tree_to_tokens(section_tree) if section.sub_sections == []]

    state = {
        "raw_text": section_tree.title_parsed + "\n" + section_tree.content_parsed,
        "section_tree": section_tree,
        "current_depth": 0,
        "max_depth": 1,
        "sections_to_process": section_tree_no_sub_sections,
        "processed_sections": []
    }
    result = asyncio.run(cut_deeper_node(state))

    remove_circular_references(section_tree)

    with open('section_tree_cut_by_1_times_depth.json', 'w') as f:
        json.dump(section_tree.model_dump(), f, indent=4, ensure_ascii=False)
