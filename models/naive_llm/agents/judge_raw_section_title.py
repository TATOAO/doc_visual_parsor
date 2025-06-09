from models.utils.schemas import Section
from models.utils.llm import get_llm_client
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

async def whether_raw_section_title(section: Section) -> bool:
    llm_client = get_llm_client(model="qwen-max-latest")
    prompt = """
你是个文档理解专家，现在有一个实习生给了一个文档的章节标题和内容。
你需要判断该章节标题和原文的信息，是否是实习生总结的, 还是原文本来就有的。
如果是实习生总结的，返回true，否则返回false。如果是原文本来就有的，返回true。

输出格式：
{
    "is_raw_section_title": bool
}
    """



    text = f"实习生给的标题: {section.title}\n  原文信息: {section.title_parsed}\n {section.content_parsed}"
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text)
    ]
    chain = llm_client | JsonOutputParser()

    response = await chain.ainvoke(messages)
    return response.get("is_raw_section_title", False)


# python -m models.naive_llm.agents.judge_raw_section_title
if __name__ == "__main__":
    import asyncio

    sample_section = Section(
        title="倒签合同",
        content="",
        level=1,
        paragraph_index=0,
        style="",
        title_parsed="",
        content_parsed="""（二）合同管理常见易发问题
倒签合同或补签合同，导致合同履行时，双方权利义务关系不明确，存在合规风险。
合同承办人员未严格履行招投标程序或比质比价程序，可能引发廉洁合规风险。
合同承办人员为规避企业合同分级审批管理，拆分合同标的额较大的合同，导致合规风险。
合同签订前未对合同相对方资质、信用状况等进行调查或调查流于形式，可能导致合同无效或事实上无法履行。
合同签订前，未核实合同相对方委托代理人委托代理权限的，可能导致合同无效或被撤销。
合同文本必备条款缺失或不明确，可能导致合同执行困难、交易目的无法圆满实现。
合同履行过程中，双方协商对合同主要条款进行变更，但未及时签订书面补充协议的，可能导致合同后续履行争议。
合同履行过程中，未严格按照合同约定对货物验收情况进行书面记录的，发生争议时，可能因证据缺失，导致增加妥善解决合同争议难度。
合同档案管理松散，发生争议时，支撑己方的证据材料缺失，导致增加妥善解决合同争议难度。""",
    )

    asyncio.run(whether_raw_section_title(sample_section))
