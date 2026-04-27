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
    collection_name="support_docs_compression"
)

def compress_context(query, retrieved_docs, llm):
    """Use LLM to extract only relevant information"""
    compression_prompt = f"""Given this query: "{query}"

Extract ONLY the information relevant to answering this query from the following documents. Remove everything else.

Documents:
"""
    for i, doc in enumerate(retrieved_docs, 1):
        compression_prompt += f"\nDocument {i}:\n{doc.page_content}\n"

    compression_prompt += "\nProvide only the relevant facts, keeping each under 50 words:"

    response = llm.invoke(compression_prompt)
    return response.content

# Test it
query = "What's the refund window for electronic goods?"
docs = vectorstore.similarity_search(query, k=5)

print("Original context length:", sum(len(d.page_content) for d in docs), "chars")

compressed = compress_context(query, docs, llm)
print("Compressed context length:", len(compressed), "chars")
print("\nCompressed context:")
print(compressed)
