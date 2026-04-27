# section3_naive_problems.py
from config.llm_config import get_embeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

sample_documents = [
    """Customer Support Best Practices - Chapter 1

    When handling refund requests, always verify the purchase date first. Our policy allows
    refunds within 30 days of purchase for most items. Electronic goods have a 14-day window.
    Items on sale or clearance are final sale and cannot be refunded.

    Process:
    1. Check purchase date in the order system
    2. Verify item eligibility (not clearance, within window)
    3. Ask for reason (helps product team improve)
    4. Process refund to original payment method
    5. Send confirmation email within 24 hours

    Common issues: Customers often don't have receipts. You can look up orders by email
    and order number. If unable to verify, escalate to a supervisor.
    """,

    """Customer Support Best Practices - Chapter 2

    Shipping inquiries are the most common support ticket type. Most shipping delays are
    carrier-related and outside our control, but customers don't care whose fault it is.

    Response template for delayed shipments:
    "I understand how frustrating delays are. Let me check your order status...
    [check tracking]... I can see it's currently in transit. Expected delivery is [date].
    If it doesn't arrive by then, contact us and we'll send a replacement immediately."

    For lost packages (tracking shows delivered but customer denies receipt):
    1. Check if it was signed for (if yes, verify signature)
    2. Ask customer to check with neighbours and building concierge
    3. File carrier claim if still missing after 48 hours
    4. Send replacement immediately (don't wait for claim resolution)
    """,
]

embeddings = get_embeddings()

# Use LangChain's text splitter (production-ready chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust based on your needs
    chunk_overlap=50,  # Overlap prevents information loss at boundaries
    length_function=len,
)

# Combine our sample documents
full_text = "\n\n".join(sample_documents)

# Split and create chunks
chunks = text_splitter.split_text(full_text)
print(f"Created {len(chunks)} chunks")
print(f"Average chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

# Create vector database
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="support_docs"
)

print("\n✓ Vector database created and populated")

# Ambiguous query (multiple meanings)
ambiguous_query = "What's the return policy?"

# Could mean:
# 1. Refund policy (product returns)
# 2. Shipping return procedures
# 3. Return authorization process

retrieved = vectorstore.similarity_search(ambiguous_query, k=3)
print(f"Query: {ambiguous_query}\n")
print("Retrieved chunks:")
for i, doc in enumerate(retrieved, 1):
    print(f"{i}. {doc.page_content[:150]}...")


# section3_naive_problems.py continued
from config.llm_config import get_llm

# Instead of searching with the vague query, expand it
def expand_query(original_query, llm):
    expansion_prompt = f"""Given this user question: "{original_query}"

Generate 3 expanded variations that capture different possible intents:
1. [First variation]
2. [Second variation]
3. [Third variation]

Keep each variation concise (under 15 words)."""

    response = llm.invoke(expansion_prompt)
    return response.content

llm = get_llm()

# Expand the ambiguous query
original = "What's the return policy?"
expanded = expand_query(original, llm)
print("Original query:", original)
print("\nExpanded queries:")
print(expanded)

# Now search with all variations and combine results
# (In production, you'd deduplicate and rank the combined results)


# section3_naive_problems.py continued
from langchain_core.documents import Document

# Recreate documents with metadata
docs_with_metadata = [
    Document(
        page_content=chunks[0],
        metadata={"source": "refund_policy", "category": "returns", "date": "2025-01"}
    ),
    Document(
        page_content=chunks[1],
        metadata={"source": "refund_policy", "category": "returns", "date": "2025-01"}
    ),
    Document(
        page_content=chunks[2],
        metadata={"source": "shipping_guide", "category": "shipping", "date": "2025-02"}
    ),
]

# Create new vectorstore with metadata
vectorstore_with_meta = Chroma.from_documents(
    documents=docs_with_metadata,
    embedding=embeddings,
    collection_name="support_docs_meta"
)

# Now we can filter during retrieval
query = "How do I handle shipping delays?"

# Filter to only shipping-related documents
results = vectorstore_with_meta.similarity_search(
    query,
    k=3,
    filter={"category": "shipping"}  # Only search shipping docs
)

print(f"Query: {query}")
print(f"Retrieved {len(results)} chunks from 'shipping' category only")

# section3_naive_problems.py continued

# Step 1: Retrieve more chunks than you need (cast a wide net)
query = "Customer says package was delivered but they never received it. What do I do?"
candidates = vectorstore.similarity_search(query, k=8)  # Get 8, will narrow to 3

# Step 2: Use LLM to re-rank
def rerank_with_llm(query, candidates, llm, top_k=3):
    rerank_prompt = f"""Given this query: "{query}"

Rank these document chunks by relevance (1 = most relevant):

"""
    for i, doc in enumerate(candidates):
        rerank_prompt += f"\nChunk {i+1}:\n{doc.page_content}\n"
    
    rerank_prompt += f"\nProvide ONLY the chunk numbers in order of relevance, separated by commas (e.g., '3,1,5'):"
    
    response = llm.invoke(rerank_prompt)
    # Parse the response to get rankings
    # In production, use a structured output format
    return response.content

rankings = rerank_with_llm(query, candidates, llm)
print(f"LLM rankings: {rankings}")
print("\nThis re-ranking helps when semantic similarity isn't enough.")
