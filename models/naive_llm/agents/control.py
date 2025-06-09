from models.utils.schemas import Section, Positions
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
import asyncio
import logging

logger = logging.getLogger(__name__)

from .section_cutter import cut_section_tree
from .judger_need_deeper_cut import whether_need_deeper_cut
from .judge_raw_section_title import whether_raw_section_title
from ..helpers import flatten_section_tree_to_tokens, remove_circular_references


class SectionCuttingState(TypedDict):
    """State for the section cutting workflow"""
    raw_text: str
    section_tree: Optional[Section]
    current_depth: int
    max_depth: int
    sections_to_process: List[Section]
    processed_sections: List[Section]


async def initial_cut_node(state: SectionCuttingState) -> SectionCuttingState:
    """Initial cutting of the document into sections"""
    logger.info(f"Starting initial cut - text length: {len(state['raw_text'])} characters")
    
    section_tree = await cut_section_tree(state["raw_text"], max_depth=3)
    
    # Get leaf sections for processing
    leaf_sections = [
        section for section in flatten_section_tree_to_tokens(section_tree) 
        if section.sub_sections == []
    ]
    
    logger.info(f"Initial cut completed - found {len(leaf_sections)} leaf sections")
    
    return {
        **state,
        "section_tree": section_tree,
        "sections_to_process": leaf_sections,
        "processed_sections": []
    }


async def validate_titles_node(state: SectionCuttingState) -> SectionCuttingState:
    """Validate that section titles are from raw text, not LLM-generated"""
    
    async def validate_section_title(section: Section) -> Optional[Section]:
        """Validate if a section title is from raw text"""
        if await whether_raw_section_title(section):
            return section
        return None
    
    # Validate all sections to process concurrently
    validation_tasks = [validate_section_title(section) for section in state["sections_to_process"]]
    validation_results = await asyncio.gather(*validation_tasks)
    
    # Filter out sections with LLM-generated titles (None results)
    valid_sections = [section for section in validation_results if section is not None]
    
    return {
        **state,
        "sections_to_process": valid_sections
    }


async def judge_sections_node(state: SectionCuttingState) -> SectionCuttingState:
    """Judge which sections need deeper cutting concurrently"""
    # Create async tasks for judging each section
    async def judge_section(section: Section) -> Optional[Section]:
        if await whether_need_deeper_cut(section):
            return section
        return None
    
    # Run all judgments concurrently
    judgment_tasks = [judge_section(section) for section in state["sections_to_process"]]
    judgment_results = await asyncio.gather(*judgment_tasks)
    
    # Filter out None results
    sections_needing_cut = [section for section in judgment_results if section is not None]
    
    return {
        **state,
        "sections_to_process": sections_needing_cut
    }


async def cut_deeper_node(state: SectionCuttingState) -> SectionCuttingState:
    """
    Cut sections deeper and fix their positions concurrently

    Output:
    {
        "section_tree": Section,
        "sections_to_process": List[Section],
        "processed_sections": List[Section],
        "current_depth": int,
        "max_depth": int
    }
    """
    
    async def process_section(leaf_section: Section) -> Section:
        """Process a single section concurrently"""
        # Cut the section deeper
        new_section_tree = await cut_section_tree(
            leaf_section.title_parsed + "\n" + leaf_section.content_parsed,
            max_depth= 1
        )

        remove_circular_references(new_section_tree)
        leaf_section.sub_sections = new_section_tree.sub_sections
        
        # Fix the positions of the new section tree
        for section in flatten_section_tree_to_tokens(new_section_tree):
            section.title_position = Positions.from_text(
                section.title_position.text_position.start + leaf_section.title_position.text_position.start, 
                section.title_position.text_position.end + leaf_section.title_position.text_position.start
            )
            section.content_position = Positions.from_text(
                section.content_position.text_position.start + leaf_section.content_position.text_position.start, 
                section.content_position.text_position.end + leaf_section.content_position.text_position.start
            )

            section.level = leaf_section.level + section.level

        # Fix the parent section of the new section tree
        for section in flatten_section_tree_to_tokens(new_section_tree):
            section.parent_section = leaf_section
        
        return leaf_section
    
    # Process all sections concurrently
    processing_tasks = [process_section(section) for section in state["sections_to_process"]]
    processed_sections = await asyncio.gather(*processing_tasks)
    
    # Get new leaf sections for next iteration
    new_leaf_sections = []
    for processed_section in processed_sections:
        new_leaves = [
            section for section in flatten_section_tree_to_tokens(processed_section)
            if section.sub_sections == [] and section != processed_section
        ]
        new_leaf_sections.extend(new_leaves)
    
    return {
        **state,
        "sections_to_process": new_leaf_sections,
        "processed_sections": state["processed_sections"] + processed_sections,
        "current_depth": state["current_depth"] + 1
    }


def should_continue_cutting(state: SectionCuttingState) -> str:
    """Decide whether to continue cutting or end the workflow"""
    # Stop if we've reached max depth
    if state["current_depth"] >= state["max_depth"]:
        return "end"
    
    # Stop if there are no sections to process
    if not state["sections_to_process"]:
        return "end"
    
    return "continue"


def should_continue_after_validation(state: SectionCuttingState) -> str:
    """Decide whether to continue after title validation"""
    # If no valid sections remain, end the workflow
    if not state["sections_to_process"]:
        return "end"
    
    return "continue"


def build_section_cutting_graph() -> StateGraph:
    """Build the LangGraph workflow for section cutting with title validation"""
    
    # Create the graph
    workflow = StateGraph(SectionCuttingState)
    
    # Add nodes
    workflow.add_node("initial_cut", initial_cut_node)
    workflow.add_node("validate_initial_titles", validate_titles_node)
    workflow.add_node("judge_sections", judge_sections_node)
    workflow.add_node("cut_deeper", cut_deeper_node)
    workflow.add_node("validate_deeper_titles", validate_titles_node)
    
    # Add edges
    workflow.add_edge(START, "initial_cut")
    workflow.add_edge("initial_cut", "validate_initial_titles")
    
    # Add conditional edge after initial title validation
    workflow.add_conditional_edges(
        "validate_initial_titles",
        should_continue_after_validation,
        {
            "continue": "judge_sections",
            "end": END
        }
    )
    
    # Add conditional edge for the recursive loop
    workflow.add_conditional_edges(
        "judge_sections",
        should_continue_cutting,
        {
            "continue": "cut_deeper",
            "end": END
        }
    )
    
    # After cutting deeper, validate the new titles
    workflow.add_edge("cut_deeper", "validate_deeper_titles")
    
    # After validating deeper titles, decide whether to continue or end
    workflow.add_conditional_edges(
        "validate_deeper_titles",
        should_continue_after_validation,
        {
            "continue": "judge_sections",
            "end": END
        }
    )
    
    return workflow.compile()


async def control_section_cutter_async(raw_text: str, max_depth: int = 3) -> Section:
    """
    Control the section cutter using LangGraph with recursive depth limiting and async concurrency
    """
    
    # Initialize the workflow
    workflow = build_section_cutting_graph()
    
    # Initial state
    initial_state: SectionCuttingState = {
        "raw_text": raw_text,
        "section_tree": None,
        "current_depth": 0,
        "max_depth": max_depth,
        "sections_to_process": [],
        "processed_sections": []
    }
    
    # Run the workflow asynchronously
    final_state = await workflow.ainvoke(initial_state)
    
    return final_state["section_tree"]


def control_section_cutter(raw_text: str, max_depth: int = 3) -> Section:
    """
    Synchronous wrapper for the async control_section_cutter_async function
    """
    return asyncio.run(control_section_cutter_async(raw_text, max_depth))


async def control_section_cutter_streaming_async(raw_text: str, max_depth: int = 3):
    """
    Streaming version that yields section trees at each depth level
    
    Yields:
        tuple[int, Section]: (current_depth, section_tree) at each processing level
    """
    from copy import deepcopy
    
    # Initialize state
    current_state: SectionCuttingState = {
        "raw_text": raw_text,
        "section_tree": None,
        "current_depth": 0,
        "max_depth": max_depth,
        "sections_to_process": [],
        "processed_sections": []
    }
    
    # Step 1: Initial cut
    current_state = await initial_cut_node(current_state)
    
    # Yield initial result (depth 0)
    section_tree_copy = deepcopy(current_state["section_tree"])
    remove_circular_references(section_tree_copy)
    yield (0, section_tree_copy)
    
    # Step 2: Validate initial titles
    current_state = await validate_titles_node(current_state)
    
    # Check if we should continue after validation
    if not current_state["sections_to_process"]:
        return
    
    # Main processing loop for deeper levels
    while current_state["current_depth"] < current_state["max_depth"] and current_state["sections_to_process"]:
        # Step 3: Judge which sections need deeper cutting
        current_state = await judge_sections_node(current_state)
        
        # If no sections need cutting, break
        if not current_state["sections_to_process"]:
            break
            
        # Step 4: Cut deeper
        current_state = await cut_deeper_node(current_state)
        
        # Step 5: Validate deeper titles
        current_state = await validate_titles_node(current_state)
        
        # Yield result after this depth level
        section_tree_copy = deepcopy(current_state["section_tree"])
        remove_circular_references(section_tree_copy)
        yield (current_state["current_depth"], section_tree_copy)


def control_section_cutter_streaming(raw_text: str, max_depth: int = 3):
    """
    Synchronous wrapper for the async streaming section cutter
    
    Yields:
        tuple[int, Section]: (current_depth, section_tree) at each processing level
    """
    async def _async_generator():
        async for depth, section_tree in control_section_cutter_streaming_async(raw_text, max_depth):
            yield depth, section_tree
    
    # Run async generator in sync context
    async def _run_async_gen():
        results = []
        async for depth, section_tree in control_section_cutter_streaming_async(raw_text, max_depth):
            results.append((depth, section_tree))
        return results
    
    results = asyncio.run(_run_async_gen())
    for depth, section_tree in results:
        yield depth, section_tree


# python -m models.naive_llm.agents.control
if __name__ == "__main__":
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # test control flow
    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    raw_text = extract_docx_content(docx_path)
    
    print(f"Document length: {len(raw_text)} characters")
    print(f"First 500 characters: {raw_text[:500]}...")
    
    # Check if text length might be causing issues
    if len(raw_text) > 50000:
        print("⚠️  WARNING: Document is very long, this might cause timeout issues")
        print("Consider processing in smaller chunks or using a shorter max_depth")
    
    try:
        # Original non-streaming approach
        print("\n" + "="*50)
        print("STARTING ORIGINAL APPROACH")
        print("="*50)
        
        section_tree = control_section_cutter(raw_text, max_depth=2)
        from models.naive_llm.helpers import remove_circular_references

        remove_circular_references(section_tree)
        import json
        with open('section_tree_cut_by_3_times_depth.json', 'w') as f:
            json.dump(section_tree.model_dump(), f, indent=4, ensure_ascii=False)
        
        print("✅ Original approach completed successfully")
        
        print("\n" + "="*50)
        print("STREAMING EXAMPLE:")
        print("="*50)
        
        # New streaming approach - get results depth by depth
        for depth, section_tree_at_depth in control_section_cutter_streaming(raw_text, max_depth=2):
            print(f"\n--- Depth {depth} Results ---")
            print(f"Number of sections: {len([s for s in flatten_section_tree_to_tokens(section_tree_at_depth)])}")
            
            # Save each depth level to separate files
            section_tree_copy = section_tree_at_depth
            remove_circular_references(section_tree_copy)
            with open(f'section_tree_depth_{depth}.json', 'w') as f:
                json.dump(section_tree_copy.model_dump(), f, indent=4, ensure_ascii=False)
            
            print(f"Saved results to section_tree_depth_{depth}.json")
        
        print("\n✅ Streaming approach completed successfully")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {str(e)}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify your QWEN API key and base URL")
        print("3. Try with a shorter document or lower max_depth")
        print("4. The document might be too long - consider chunking")
        
        # Print some debugging info
        print(f"\nDebugging info:")
        print(f"- Document length: {len(raw_text)} characters")
        print(f"- Max recommended length: 50,000 characters")
        print(f"- Consider reducing max_depth from 2 to 1 for testing")
        
        raise


