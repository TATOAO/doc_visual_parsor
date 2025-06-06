import dotenv
import os
import langchain_openai

dotenv.load_dotenv()

LLM_API_KEY = os.getenv("QWEN_KEY")
LLM_BASE_URL = os.getenv("QWEN_BASE_URL")

def get_llm_client(**kwargs):
    return langchain_openai.ChatOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, **kwargs)