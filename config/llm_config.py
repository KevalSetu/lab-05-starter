import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
import requests

load_dotenv()
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama

LOCAL_EMBEDDING_MODELS = {
    "nomic": "nomic-embed-text",
}

REMOTE_EMBEDDING_MODELS = {
    "text-embedding-3-small": "text-embedding-3-small",
}

def get_embeddings(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> Embeddings:
    env_provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
    if provider is None:
        provider = env_provider

    # Auto-fallback: prefer openai if key available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and provider == "local":
        print("[INFO] OpenAI key found, switching from local to openai")
        provider = "openai"

    if model_name is None:
        if provider == env_provider:
            model_name = os.getenv("EMBEDDINGS_MODEL")

    if provider == "local":
        if model_name is None:
            model_name = "nomic"

        model_id = LOCAL_EMBEDDING_MODELS.get(model_name, model_name)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Health check Ollama
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            response.raise_for_status()
            print(f"[OK] Ollama running at {base_url}")
        except requests.exceptions.RequestException:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("[WARN] Ollama not available, falling back to OpenAI")
                model_id = REMOTE_EMBEDDING_MODELS.get(model_name, "text-embedding-3-small")
                return OpenAIEmbeddings(model=model_id, api_key=api_key, **kwargs)
            else:
                raise RuntimeError(
                    f"Ollama not running at {base_url}. Start with 'ollama serve' or set OPENAI_API_KEY."
                )

        return OllamaEmbeddings(
            model=model_id,
            base_url=base_url,
            **kwargs
        )

    elif provider == "openai":
        if model_name is None:
            model_name = "text-embedding-3-small"

        model_id = REMOTE_EMBEDDING_MODELS.get(model_name, model_name)
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for OpenAI provider"
            )

        return OpenAIEmbeddings(
            model=model_id,
            api_key=api_key,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Must be 'local' or 'openai'"
        )


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
):
    env_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider is None:
        provider = env_provider

    # Auto-fallback: prefer openai if key available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and provider == "local":
        print("[INFO] OpenAI key found, switching from local to openai")
        provider = "openai"

    if model_name is None:
        if provider == env_provider:
            model_name = os.getenv("LLM_MODEL")

    if provider == "local":
        if model_name is None:
            model_name = "llama2"

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Health check Ollama
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            response.raise_for_status()
            print(f"[OK] Ollama running at {base_url}")
        except requests.exceptions.RequestException:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                print("[WARN] Ollama not available, falling back to OpenAI")
                if model_name is None:
                    model_name = "gpt-3.5-turbo"
                return ChatOpenAI(model=model_name, api_key=api_key, **kwargs)
            else:
                raise RuntimeError(
                    f"Ollama not running at {base_url}. Run 'ollama serve' & 'ollama pull {model_name}' or set OPENAI_API_KEY."
                )

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            **kwargs
        )

    elif provider == "openai":
        if model_name is None:
            model_name = "gpt-3.5-turbo"

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for OpenAI provider"
            )

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Must be 'local' or 'openai'"
        )
