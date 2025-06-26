from doc_chunking.schemas.schemas import Section
from doc_chunking.utils.llm import get_llm_client
from ..helpers import generate_section_tree_from_tokens, flatten_section_tree_to_tokens
from typing import AsyncGenerator, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)

# Single prompt template for all LLM operations
SECTION_ANALYSIS_PROMPT = """你是一个专业的文档结构分析助手，能够将文本解析为层次化的章节树结构。

任务说明：
请分析给定的文本内容，识别其中的章节结构，并使用特殊标记来分别标示章节标题和内容的开始和结束位置。

**注意**
1. 不要总结，不要概括，不要添加任何解释， 不要自己生成标题。
2. 内容可以适当省略，但是标题必须完整。

特殊标记格式：
- 第一级章节标题：<start-section-title-1> 标题内容 <end-section-title-1>
- 第一级章节内容：<start-section-content-1> 内容... <end-section-content-1>
- 第一级章节的第一个子章节标题：<start-section-title-1-1> 子标题 <end-section-title-1-1>
- 第一级章节的第一个子章节内容：<start-section-content-1-1> 子内容... <end-section-content-1-1>
- 第一级章节的第二个子章节标题：<start-section-title-1-2> 子标题 <end-section-title-1-2>
- 第一级章节的第二个子章节内容：<start-section-content-1-2> 子内容... <end-section-content-1-2>
- 第二级章节标题：<start-section-title-2> 标题内容 <end-section-title-2>
- 第二级章节内容：<start-section-content-2> 内容... <end-section-content-2>
- 依此类推...

输出要求：
1. 保持原文内容不变，只在适当位置插入特殊标记
2. 准确识别章节标题和正文内容，分别用不同的标记包围
3. 章节标题用 <start-section-title-X> 和 <end-section-title-X> 包围
4. 章节内容用 <start-section-content-X> 和 <end-section-content-X> 包围
5. 确保标记的嵌套关系正确反映文档的层次结构
6. 每个开始标记都必须有对应的结束标记
7. **重要**：绝对不能在标记外留下任何内容。所有文本都必须包含在 <start-section-title-X>...<end-section-title-X> 或 <start-section-content-X>...<end-section-content-X> 标记内
8. **严禁**：在 <end-section-content-X> 和 <start-section-title-Y> 之间留下任何文本内容

示例输出格式：
<start-section-title-1>
第一章 标题
<end-section-title-1>
<start-section-content-1>
这里是第一章的引言内容...

<start-section-title-1-1>
1.1 子标题
<end-section-title-1-1>
<start-section-content-1-1>
这里是1.1节的具体内容...
<end-section-content-1-1>

<start-section-title-1-2>
1.2 另一个子标题
<end-section-title-1-2>
<start-section-content-1-2>
这里是1.2节的具体内容...
<end-section-content-1-2>

<end-section-content-1>

<start-section-title-2>
第二章 标题
<end-section-title-2>
<start-section-content-2>
这里是第二章的内容...
<end-section-content-2>

错误示例（绝对禁止）：
<start-section-content-1-1>
这里是内容...
<end-section-content-1-1>

这里有遗漏的内容！！！ ← 这种情况绝对不允许

<start-section-title-1-2>
下一个标题
<end-section-title-1-2>

请分析以下文本并添加相应的章节标记："""


async def get_tagged_text_by_llm_streaming(text: str, progress_callback=None) -> AsyncGenerator[str, None]:
    """
    Streaming version that yields partial results as they come in
    
    Args:
        text: Input text to process
        progress_callback: Optional callback function to handle progress updates
    
    Yields:
        str: Partial responses as they stream in
    """
    try:
        logger.info(f"Starting LLM processing - text length: {len(text)}")
        
        # Use longer timeout for longer texts
        timeout = min(600, max(180, len(text) // 100))  # 3-10 minutes based on text length
        llm_client = get_llm_client(model="qwen-max-latest", timeout=timeout, max_retries=2)
        
        from langchain_core.messages import SystemMessage, HumanMessage
        
        messages = [
            SystemMessage(content=SECTION_ANALYSIS_PROMPT),
            HumanMessage(content=text)
        ]
        
        full_response = ""
        async for chunk in llm_client.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                
                # Call progress callback if provided
                if progress_callback:
                    await progress_callback(chunk.content, full_response)
                
                # Yield the chunk for real-time processing
                yield chunk.content
        
        logger.info(f"LLM processing completed - total length: {len(full_response)}")
        
    except Exception as e:
        logger.error(f"LLM processing failed: {str(e)}")
        raise


async def cut_section_tree_streaming(raw_text: str, max_depth: int = -1, progress_callback=None) -> AsyncGenerator[Tuple[bool, Section], None]:
    """
    Streaming version of cut_section_tree that yields partial results
    
    Args:
        raw_text: Input text to process
        max_depth: Maximum depth for section tree
        progress_callback: Optional callback for progress updates
    
    Yields:
        tuple: (is_complete: bool, section_tree: Section or None)
    """
    
    # Yield initial status
    yield (False, None)
    
    full_llm_result = ""
    yield_length_limit = 1000
    
    # Stream the LLM processing every 1000 characters
    async for chunk in get_tagged_text_by_llm_streaming(raw_text, progress_callback):
        full_llm_result += chunk

        print(chunk, end="", flush=True)

        # Try to parse partial results if we have enough content
        if len(full_llm_result) > yield_length_limit and '<end-section-title-' in full_llm_result:
            try:
                # Try to create a partial section tree
                partial_section_tree = generate_section_tree_from_tokens(full_llm_result, raw_text, max_depth)
                
                # Yield partial results
                yield_length_limit += 1000
                yield (False, partial_section_tree)
                
            except Exception as e:
                # If parsing fails, continue streaming
                logger.debug(f"Partial parsing failed: {e}")
                continue
    
    # Final processing
    try:
        final_section_tree = generate_section_tree_from_tokens(full_llm_result, raw_text, max_depth)

        for section in flatten_section_tree_to_tokens(final_section_tree):
            section.title_parsed = raw_text[section.title_position.text_position.start:section.title_position.text_position.end]
            section.content_parsed = raw_text[section.content_position.text_position.start:section.content_position.text_position.end]

        # Yield final complete result
        yield (True, final_section_tree)
        
    except Exception as e:
        logger.error(f"Final parsing failed: {e}")
        yield (True, None)


async def cut_section_tree(raw_text: str, max_depth: int = -1) -> Section:
    """
    Non-streaming wrapper for cut_section_tree_streaming
    Returns only the final complete result
    """
    async for is_complete, section_tree in cut_section_tree_streaming(raw_text, max_depth):
        if is_complete and section_tree is not None:
            return section_tree
    
    raise ValueError("Failed to get section tree from streaming cut")


# python -m models.naive_llm.agents.section_cutter
if __name__ == "__main__":
    import asyncio
    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    text = extract_docx_content(docx_path)
    
    async def main():

        # streaming version
        chunk_index = 0
        from ..helpers import remove_circular_references
        import json
        async for is_complete, section_tree in cut_section_tree_streaming(text, max_depth=3):
            print(f"chunk_index: {chunk_index}, is_complete: {is_complete}")
            if section_tree is not None:
                remove_circular_references(section_tree)
                with open(f'section_tree_cut_{chunk_index}.json', 'w') as f:
                    json.dump(section_tree.model_dump(), f, indent=4, ensure_ascii=False)
            else:
                print("section_tree is None")
            chunk_index += 1

    
    asyncio.run(main())