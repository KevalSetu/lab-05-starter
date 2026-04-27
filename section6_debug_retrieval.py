from config.llm_config import get_embeddings, get_llm
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = get_embeddings()
llm = get_llm(temperature=0)

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text("\n\n".join(sample_documents))
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="support_docs_debug"
)

def build_rag_prompt(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return f"""You are a customer support assistant. Use the following support documentation to answer the question.

Documentation:
{context}

Question: {query}

Provide a clear, helpful answer based on the documentation. If the documentation doesn't contain enough information, say so.

Answer:"""

# ── Exercise 6.1: Debugging Poor Retrieval ────────────────────────────────────

def diagnose_retrieval(query, vectorstore, k=5):
    """Debug why retrieval might be failing"""
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"Query: {query}\n")
    print("Retrieval diagnosis:")
    print("=" * 60)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\nRank {i} (similarity: {score:.3f}):")
        print(f"Content: {doc.page_content[:150]}...")

        query_terms = set(query.lower().split())
        doc_terms = set(doc.page_content.lower().split())
        overlap = query_terms.intersection(doc_terms)

        print(f"Term overlap: {len(overlap)} words ({overlap})")

        if score < 0.5:
            print("[WARN] Low similarity score - might not be relevant")
        if len(overlap) < 2:
            print("[WARN] Very little term overlap - semantic match only")

print("\n" + "#"*60)
print("# Exercise 6.1: Debugging Poor Retrieval")
print("#"*60)

poor_query = "help me with stuff"
diagnose_retrieval(poor_query, vectorstore)

print()
good_query = "How do I process a refund for an item purchased 20 days ago?"
diagnose_retrieval(good_query, vectorstore)

# ── Exercise 6.2: Edge Cases ──────────────────────────────────────────────────

print("\n" + "#"*60)
print("# Exercise 6.2: Handling Edge Cases")
print("#"*60)

edge_case_queries = [
    "What about items returned after 90 days?",
    "Can I get a refund in cash if I paid by card?",
    "hjkl asdfg qwerty",
    "What's the capital of France?",
    "",
]

for query in edge_case_queries:
    if not query.strip():
        print("\nEmpty query - skipping")
        continue

    print(f"\nEdge case: {query}")

    try:
        docs = vectorstore.similarity_search(query, k=3)
        response = llm.invoke(build_rag_prompt(query, docs))
        print(f"Response: {response.content[:100]}...")

        if any(phrase in response.content.lower() for phrase in
               ["i don't have", "not in the documentation", "unable to find"]):
            print("[OK] Model appropriately admits lack of knowledge")
        else:
            print("[WARN] Check for potential hallucination")

    except Exception as e:
        print(f"Error: {e}")

    print("-" * 60)

# ── Exercise 6.3: Multi-Hop Reasoning ────────────────────────────────────────

print("\n" + "#"*60)
print("# Exercise 6.3: Multi-Hop Reasoning")
print("#"*60)

multi_hop_doc = """
Shipping costs:
- Standard shipping: £5 for orders under £50
- Free shipping for orders over £50

Refund policy:
- Full refunds include original purchase price
- Shipping costs are NOT refunded unless item was defective
"""

vectorstore.add_texts([multi_hop_doc])

multi_hop_query = "If I bought a £45 item and want to return it, how much will I get back?"

docs = vectorstore.similarity_search(multi_hop_query, k=3)
response = llm.invoke(build_rag_prompt(multi_hop_query, docs))

print("Multi-hop query:", multi_hop_query)
print("\nResponse:", response.content)
print("\nDid it correctly reason across both pieces of information?")

# ── Exercise 6.4: Cold Start Problem ─────────────────────────────────────────

print("\n" + "#"*60)
print("# Exercise 6.4: Cold Start Problem")
print("#"*60)

sparse_vectorstore = Chroma.from_texts(
    texts=["Company policy: We process refunds within 5-7 business days"],
    embedding=embeddings,
    collection_name="sparse_docs"
)

query = "What's your return policy for electronics?"

docs = sparse_vectorstore.similarity_search(query, k=3)

print(f"Query: {query}")
print(f"Retrieved {len(docs)} docs (asked for 3)")
print("\nRetrieved content:")
for doc in docs:
    print(f"- {doc.page_content}")

response = llm.invoke(build_rag_prompt(query, docs))
print(f"\nResponse with sparse knowledge base:")
print(response.content)

fallback_prompt = f"""The knowledge base doesn't contain specific information about this query: "{query}"

Provide a general answer, but clearly state you're not using company-specific information:"""

fallback_response = llm.invoke(fallback_prompt)
print(f"\nFallback response:")
print(fallback_response.content)
