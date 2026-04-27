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
    collection_name="support_docs_instructions"
)

query = "What should I do if a customer doesn't have a receipt?"
docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([doc.page_content for doc in docs])

# Weak instructions
weak_prompt = f"""Context: {context}

Question: {query}

Answer:"""

# Strong instructions
strong_prompt = f"""You are a customer support assistant. Answer the question using ONLY the provided documentation.

Documentation:
{context}

Question: {query}

Instructions:
- Base your answer on the documentation above
- If the documentation doesn't contain enough information, say "I don't have that information in our support docs"
- Be specific and cite relevant policies
- Keep your answer concise (under 100 words)

Answer:"""

# Compare
weak_response = llm.invoke(weak_prompt)
strong_response = llm.invoke(strong_prompt)

print("Weak instructions response:")
print(weak_response.content)
print("\n" + "="*60 + "\n")
print("Strong instructions response:")
print(strong_response.content)
