from models.utils.schemas import Section, Positions
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
import asyncio

from .section_cutter import cut_section_tree
from .judger_need_deeper_cut import whether_need_deeper_cut
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
    section_tree = await cut_section_tree(state["raw_text"], max_depth=3)
    
    # Get leaf sections for processing
    leaf_sections = [
        section for section in flatten_section_tree_to_tokens(section_tree) 
        if section.sub_sections == []
    ]
    
    return {
        **state,
        "section_tree": section_tree,
        "sections_to_process": leaf_sections,
        "processed_sections": []
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


def build_section_cutting_graph() -> StateGraph:
    """Build the LangGraph workflow for section cutting"""
    
    # Create the graph
    workflow = StateGraph(SectionCuttingState)
    
    # Add nodes
    workflow.add_node("initial_cut", initial_cut_node)
    workflow.add_node("judge_sections", judge_sections_node)
    workflow.add_node("cut_deeper", cut_deeper_node)
    
    # Add edges
    workflow.add_edge(START, "initial_cut")
    workflow.add_edge("initial_cut", "judge_sections")
    
    # Add conditional edge for the recursive loop
    workflow.add_conditional_edges(
        "judge_sections",
        should_continue_cutting,
        {
            "continue": "cut_deeper",
            "end": END
        }
    )
    
    # Loop back from cut_deeper to judge_sections
    workflow.add_edge("cut_deeper", "judge_sections")
    
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


# python -m models.naive_llm.agents.control
if __name__ == "__main__":


    # # test control flow
    # from backend.docx_processor import extract_docx_content
    # docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    # raw_text = extract_docx_content(docx_path)
    # section_tree = control_section_cutter(raw_text, max_depth=3)
    # from ..helpers import remove_circular_references
    # remove_circular_references(section_tree)
    # import json
    # with open('section_tree_cut_by_3_times_depth.json', 'w') as f:
    #     json.dump(section_tree.model_dump(), f, indent=4, ensure_ascii=False)


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

