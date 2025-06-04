from .utils.llm import get_model

model = get_model()

SYSTEM_PROMPT = """
You are a helpful assistant that can parse documents.  

You will be given a document and you will need to parse it into a structured format.

"""


def parse_document(document: str) -> str:
    response = model.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": document}
        ]
    )
    return response.choices[0].message.content