from models.layout_structuring.title_structure_builder_llm.layout_displayer import display_layout
from models.schemas.layout_schemas import LayoutExtractionResult
from models.utils.llm import get_llm_client

def title_structure_builder_llm(layout_extraction_result: LayoutExtractionResult) -> str:
    llm_client = get_llm_client(model="qwen3-8b", extra_body={"enable_thinking": False})
    prompt = f"""
    你是一个可以帮助构建文档标题结构的智能助手。
    文档内容如下：
    {display_layout(layout_extraction_result)}

    输入的文本开头包含的是id, page_number, style, font, alignment, bbox 等信息，这些信息仅作参考，不需要构建到标题结构中。

    请帮我构建这个文档的标题结构。
    注意只需要构建标题结构，不需要构建其他内容。
    输出格式例如:
    1. 标题1
    1.1. 标题1.1
        1.1.1. 标题1.1.1
        1.1.2. 标题1.1.2
    1.2. 标题1.2
    2. 标题2
    2.1. 标题2.1
    2.2. 标题2.2
    2.3. 标题2.3
    """

    messages = [
        {"role": "system", "content": "你是一个可以帮助构建文档标题结构的智能助手。"},
        {"role": "user", "content": prompt}
    ]

    response = llm_client.stream(messages)
    content = ""
    
    for chunk in response:
        # Regular content
        if chunk.content:
            print(chunk.content, end="", flush=True)
            content += chunk.content
    
    return content


# python -m models.layout_structuring.title_structure_builder_llm.structurer_llm
if __name__ == "__main__":
    layout_extraction_result = LayoutExtractionResult.model_validate_json(open("./hybrid_extraction_result.json", "r").read())
    title_structure = title_structure_builder_llm(layout_extraction_result)
    # print(title_structure)