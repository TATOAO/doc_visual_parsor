from models.naive_llm.agents.control import control_section_cutter_streaming
from models.naive_llm.helpers import remove_circular_references
import json



# python -m models.naive_llm.tests.test_control_steamingly
if __name__ == "__main__":
    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    raw_text = extract_docx_content(docx_path)


    # Stream results depth by depth
    for depth, section_tree in control_section_cutter_streaming(raw_text, max_depth=2):
        print(f"Got results for depth {depth}")
        # Process the section_tree immediately
        # Save, display, or analyze the current depth results

        # Save each depth level to separate files
        section_tree_copy = section_tree
        remove_circular_references(section_tree_copy)
        with open(f'section_tree_depth_{depth}.json', 'w') as f:
            json.dump(section_tree_copy.model_dump(), f, indent=4, ensure_ascii=False)