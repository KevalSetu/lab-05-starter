import os
from typing import Optional
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv

load_dotenv()

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
    """
    Factory function to get appropriate embeddings model based on configuration.

    Args:
        provider: 'local' or 'openai' (reads from env if None)
        model_name: Model identifier (reads from env if None)
        **kwargs: Additional model-specific parameters

    Returns:
        Configured embeddings instance

    Environment Variables:
        EMBEDDINGS_PROVIDER: 'local' or 'openai'
        EMBEDDINGS_MODEL: Model name/identifier
        OPENAI_API_KEY: Required for OpenAI
        OLLAMA_BASE_URL: Optional, defaults to http://localhost:11434
    """

    env_provider = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
    if provider is None:
        provider = env_provider

    if model_name is None:
        if provider == env_provider:
            model_name = os.getenv("EMBEDDINGS_MODEL")

    # LOCAL OLLAMA EMBEDDINGS
    if provider == "local":
        if model_name is None:
            model_name = "nomic"

        model_id = LOCAL_EMBEDDING_MODELS.get(model_name, model_name)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return OllamaEmbeddings(
            model=model_id,
            base_url=base_url,
            **kwargs
        )

    # OPENAI EMBEDDINGS
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

LOCAL_MODELS = {
    "llama": "llama3:8b",
    "gemma": "gemma3:1b",
    "qwen": "qwen2.5:7b-instruct"
}

REMOTE_MODELS = {
    "gpt-4.1": "gpt-4.1",
}

def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseChatModel:
    """
    Factory function to get appropriate LLM based on configuration.
    
    Args:
        provider: 'local' or 'openai' (reads from env if None)
        model_name: Model identifier (reads from env if None)
        temperature: Model temperature setting
        **kwargs: Additional model-specific parameters
    
    Returns:
        Configured LLM instance
        
    Environment Variables:
        LLM_PROVIDER: 'local' or 'openai'
        LLM_MODEL: Model name/identifier
        OPENAI_API_KEY: Required for OpenAI
        OLLAMA_BASE_URL: Optional, defaults to http://localhost:11434
    
    Examples:
        # Use environment variables
        llm = get_llm()
        
        # Explicit configuration
        llm = get_llm(provider='local', model_name='mistral')
        llm = get_llm(provider='openai', model_name='gpt-4.1-mini')
    """
    
    # Determine provider
    env_provider = os.getenv("LLM_PROVIDER", "local").lower()
    if provider is None:
        provider = env_provider
    
    # Determine model - KEY FIX: Only read from env if provider matches env_provider
    if model_name is None:
        if provider == env_provider:
            # Use environment model when using environment provider
            model_name = os.getenv("LLM_MODEL")
        # Otherwise model_name stays None and we use defaults below
    
    # LOCAL OLLAMA MODELS
    if provider == "local":
        if model_name is None:
            model_name = "gemma"  # Default local model
        
        # Get full model identifier
        model_id = LOCAL_MODELS.get(model_name, model_name)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model_id,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    
    # OPENAI MODELS
    elif provider == "openai":
        if model_name is None:
            model_name = "gpt-4.1-mini"  # Default OpenAI model
        
        model_id = REMOTE_MODELS.get(model_name, model_name)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for OpenAI provider"
            )
        

        return ChatOpenAI(
            model=model_id,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Must be 'local' or 'openai'"
        )

def list_available_models(provider: str) -> dict:
    """List all available models for a given provider."""
    if provider == "local":
        return LOCAL_MODELS
    elif provider == "openai":
        return REMOTE_MODELS
    else:
        raise ValueError(f"Unknown provider: {provider}")


def test_llm_connection(llm: BaseChatModel) -> bool:
    """Test if LLM is properly configured and accessible."""
    try:
        response = llm.invoke("Say 'hello' if you can hear me.")
        return bool(response.content)
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
  
if __name__ == "__main__":
  llm = get_llm()
  response = llm.invoke("Say hello and confirm you're working")
  print(response.content)