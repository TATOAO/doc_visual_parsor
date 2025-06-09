import re
from models.utils.schemas import Section
from models.utils.llm import get_llm_client

"""
The general idea is use llm to understand the text and inserting special tokens to indicate the section structure.
Then use another regex to parse the tokens and generate a section tree.

The special tokens are:
<start-section-title-1>
<end-section-title-1>
<start-section-content-1>

<start-section-title-1-1>
<end-section-title-1-1>
<start-section-content-1-1>
<end-section-content-1-1>

<start-section-title-1-2>
<end-section-title-1-2>
<start-section-content-1-2>
<end-section-content-1-2>

<start-section-title-2>
<end-section-title-2>
<start-section-content-2>

<start-section-title-2-1>
<end-section-title-2-1>
<start-section-content-2-1>
<end-section-content-2-1>

<start-section-title-2-2>
<end-section-title-2-2>
<start-section-content-2-2>
<end-section-content-2-2>

<end-section-content-2>
"""



def get_section_tree_by_llm(text: str) -> Section:
    """
    using llm parsing the text into a section tree
    """
    llm_client = get_llm_client(model="qwen-max-latest")
    prompt = f"""
你是一个专业的文档结构分析助手，能够将文本解析为层次化的章节树结构。

任务说明：
请分析给定的文本内容，识别其中的章节结构，并使用特殊标记来分别标示章节标题和内容的开始和结束位置。

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

请分析以下文本并添加相应的章节标记：
"""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text)
    ]
    response = llm_client.invoke(messages)
    return response.content

# python -m models.naive_llm.__init__
if __name__ == "__main__":
    from backend.docx_processor import extract_docx_content
    docx_path = "/Users/tatoaoliang/Downloads/Work/doc_visual_parsor/tests/test_data/1-1 买卖合同（通用版）.docx"
    text = extract_docx_content(docx_path)
    section_tree = get_section_tree_by_llm(text)
    print(section_tree)