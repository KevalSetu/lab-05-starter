import os
from dotenv import load_dotenv
from llm_factory import create_llm

# Load environment variables from .env
load_dotenv()

provider = os.getenv("LLM_PROVIDER", "ollama")
model = os.getenv("LLM_MODEL")

assert provider, "provider not set"
assert model, "model not set"

# Factory creates the correct LLM
llm = create_llm(provider, model)


def test_llm_connection():
    """Test if LLM is properly configured and accessible."""
    try:
        response = llm.invoke("Say 'hello' if you can hear me.")
        print("Connection Response:", response)
        # response for remote model
        # print("Connection Response:", response.content)
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


def simple_prompt():
    prompt = "Write a short poem about the ocean."
    response = llm.invoke(prompt)

    print("Model Response:")
    print("Connection Response:", response)


if __name__ == "__main__":
    print("Using provider:", provider)
    print("Testing LLM connection...")
    test_llm_connection()

    print("Running simple prompt...")
    simple_prompt()