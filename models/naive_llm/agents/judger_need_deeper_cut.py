from models.utils.schemas import Section
from models.utils.llm import get_llm_client
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

async def whether_need_deeper_cut(section: Section) -> bool:
    llm_client = get_llm_client(model="qwen-max-latest")
    prompt = """
    你是一个有用的文章标题理解助手。
    你被给定了一个文档的章节标题和内容。
    如果内容没有更多的标题，则不需要进一步细分。
    你需要判断该章节标题是否需要进一步细分。如果需要，返回True，否则返回False。

    输出格式：
    {
        "need_deeper_cut": bool
    }
    """
    text = f"Section: {section.title}\nContent: {section.content}"
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text)
    ]
    chain = llm_client | JsonOutputParser()

    response = await chain.ainvoke(messages)
    return response.get("need_deeper_cut", False)


# python -m models.naive_llm.agents.judger_need_deeper_cut
if __name__ == "__main__":
    import asyncio

    sample_section = Section(
        title="第一章 买卖合同",
        content="共计5个合同示范文本,分别是买卖合同（通用版）、设备采购合同、产品经销合同、集采零配件买卖合同、原材料采购合同。编制主要依据是《民法典》《最高人民法院关于审理买卖合同纠纷案件适用法律问题的解释》《最高人民法院关于适用<中华人民共和国民法典>合同编通则若干问题的解释》。",
        level=1,
        paragraph_index=0,
        style="",
    )

    async def main():
        result = await whether_need_deeper_cut(sample_section)
        print(result)
    
    asyncio.run(main())