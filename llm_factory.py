import os
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI


def create_llm(provider: str, model: str):
    """
    Simple LLM factory.
    Takes a provider name explicitly instead of reading environment variables.
    """

    provider = provider.lower()

    if provider == "openai":
        return ChatOpenAI(
            model= model,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    return OllamaLLM(model=model)