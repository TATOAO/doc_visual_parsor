from .section_cutter import cut_section_tree_streaming, parsed_llm_result_into_section_tree
from models.utils.schemas import Section
from asyncio import TaskGroup


async def control_section_cutter_concurrent_async(raw_text: str, max_depth: int = 3) -> Section:

    saved_section_tree_status = {}


    try: 
        with TaskGroup() as tg:
            async for is_complete, section_tree in cut_section_tree_streaming(raw_text, max_depth):
                if section_tree.section_hash in saved_section_tree_status:
                    continue
                
                if is_complete and section_tree is not None:
                    saved_section_tree_status[section_tree.section_hash] = section_tree
                    return section_tree
                
                tg.create_task(cut_section_tree_streaming(raw_text, max_depth))
    
    except Exception as e:
        raise ValueError(f"Failed to get section tree from streaming cut: {e}")


# python -m models.naive_llm.agents.control
if __name__ == "__main__":

    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    raw_text = extract_docx_content(docx_path)

    control_section_cutter_concurrent_async()
