from dotenv import load_dotenv
load_dotenv()

from config.llm_config import get_llm, get_embeddings
import chromadb

# Test LLM
llm = get_llm()
response = llm.invoke("Confirm you're working")
print(f"[OK] LLM working: {response.content[:50]}...")

embeddings = get_embeddings()

test_embed = embeddings.embed_query("test")
print(f"[OK] Embeddings working: {len(test_embed)} dimensions")

# Test ChromaDB
client = chromadb.Client()
print("[OK] ChromaDB working")
