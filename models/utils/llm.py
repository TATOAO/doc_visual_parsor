import dotenv
import os
import langchain_openai

dotenv.load_dotenv()

LLM_API_KEY = os.getenv("QWEN_KEY")
LLM_BASE_URL = os.getenv("QWEN_BASE_URL")

def get_llm_client(timeout: int = 300, max_retries: int = 3, **kwargs):
    """
    Get LLM client with proper timeout and retry configuration
    
    Args:
        timeout: Request timeout in seconds (default: 300 = 5 minutes)
        max_retries: Maximum number of retries for failed requests
        **kwargs: Additional arguments passed to ChatOpenAI
    """
    return langchain_openai.ChatOpenAI(
        api_key=LLM_API_KEY, 
        base_url=LLM_BASE_URL,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs
    )

# python -m models.utils.llm
if __name__ == "__main__":
    print(get_llm_client(model="qwen-max-latest").invoke("Hello, how are you?"))