import asyncio
from typing import AsyncGenerator
from doc_chunking.layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from doc_chunking.schemas.layout_schemas import LayoutExtractionResult, ElementType
from doc_chunking.utils.llm import get_llm_client


system_prompt = f"""
你是一个可以帮助构建文档标题结构的智能助手。

输入的文本开头包含的是id, page_number, style, font, alignment, bbox 等信息, 这些信息可以作参考

注意有两个假设:

1. 同一个level的title应该保持相同的style信息, 样式， 例如字体大小、名称、颜色等。
2. 另外同一level的title单位应该相同， 例如 第一章、(一)、1.1.1. 等， 不要混用。

你的任务是帮我构建这个文档的标题结构。
注意只需要构建标题结构，标题需要完全匹配原文，不要添加任何其他内容。
输出格式例如:
1. 标题x
1.1. 标题xxxx
    1.1.1. 标题xxxy
    1.1.2. 标题xxyy
1.2. 标题xxx
2. 标题xxx
2.1. 标题xxx
2.2. 标题xxx
2.3. 标题xxx
"""



continue_system_prompt = f"""
你是一个可以帮助构建文档标题结构的智能助手。

输入的文本开头包含的是id, page_number, style, font, alignment, bbox 等信息, 这些信息可以作参考

注意有两个假设:

1. 同一个level的title应该保持相同的style信息, 样式， 例如字体大小、名称、颜色等。
2. 另外同一level的title单位应该相同， 例如 第一章、(一)、1.1.1. 等， 不要混用。

你的任务是构建这个文档的标题结构。

注意只需要构建标题结构，标题需要完全匹配原文，不要添加任何其他内容。
输出格式例如:
1. 标题x
1.1. 标题xxxx
    1.1.1. 标题xxxy
    1.1.2. 标题xxyy
1.2. 标题xxx
2. 标题xxx
2.1. 标题xxx
2.2. 标题xxx
2.3. 标题xxx


如果用户已经提供了部分标题结构，请继续构建。
"""



async def _collect_chunks(layout_extraction_result: LayoutExtractionResult) -> str:
    """Helper async function to collect all chunks from the async generator."""
    content = ""
    async for chunk in stream_title_structure_builder_llm(layout_extraction_result):
        # print(chunk, end="", flush=True)
        content += chunk
    return content


def title_structure_builder_llm(layout_extraction_result: LayoutExtractionResult) -> str:
    """Synchronous wrapper that runs the async generator using asyncio.run()."""
    return asyncio.run(_collect_chunks(layout_extraction_result))


async def stream_title_structure_builder_llm(layout_extraction_result: LayoutExtractionResult) -> AsyncGenerator[str, None]:
    llm_client = get_llm_client(model="qwen3-4b", extra_body={"enable_thinking": False})

    user_prompt = f"""
文档内容如下：
{display_layout(layout_extraction_result, exclude_types=[])}

你可以忽略小的标题结构， 优先构建大的标题结构。总字数不超过500字。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm_client.astream(messages)
    
    async for chunk in response:
        # Regular content
        if chunk.content:
            yield chunk.content



async def stream_title_structure_builder_llm_with_plain_text(title_structure: str, displayed_lines: str) -> AsyncGenerator[str, None]:
    llm_client = get_llm_client(model="qwen3-4b", extra_body={"enable_thinking": False})

    user_prompt = f"""
这是已有标题结构：
{title_structure}

继续构建的文档内容如下：
{displayed_lines}

你可以忽略小的标题结构， 优先构建大的标题结构。总字数不超过500字。你只需要输出新的标题结构，不需要输出任何其他内容。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm_client.astream(messages)
    
    async for chunk in response:
        # Regular content
        if chunk.content:
            yield chunk.content



# python -m models.layout_structuring.title_structure_builder_llm.structurer_llm
if __name__ == "__main__":
    layout_extraction_result = LayoutExtractionResult.model_validate_json(open("./layout_detection_result_with_runs.json", "r").read())
    title_structure = title_structure_builder_llm(layout_extraction_result)
    print(title_structure)