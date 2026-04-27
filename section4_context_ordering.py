# section4_context_ordering.py
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
    collection_name="support_docs_context_ordering"
)

query = "What should I do if a customer doesn't have a receipt?"
docs = vectorstore.similarity_search(query, k=3)
context = "\n\n".join([doc.page_content for doc in docs])

# Format 1: Context-First
context_first = f"""Here is relevant documentation:

{context}

Question: {query}

Answer:"""

# Format 2: Question-First
question_first = f"""Question: {query}

Use this documentation to answer:

{context}

Answer:"""

# Format 3: Interleaved
interleaved = f"""Question: {query}

Source 1:
{docs[0].page_content}

Source 2:
{docs[1].page_content}

Source 3:
{docs[2].page_content}

Now answer the question using the sources above:"""

# Test all three
for label, prompt in [("Context-First", context_first), 
                       ("Question-First", question_first),
                       ("Interleaved", interleaved)]:
    response = llm.invoke(prompt)
    print(f"\n{'='*60}")
    print(f"{label} Format:")
    print(f"{'='*60}")
    print(response.content)


# section4_instructions.py
# You will also need a vector database and an llm
# to be defined, same as sections 2 and 3

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


# section4_lost_in_middle.py
# You will also need a vector database and an llm
# to be defined, same as sections 2 and 3

# Create a scenario where the answer is in the middle chunk
query = "What's the equipment reimbursement amount for home office setup?"

# Assume we retrieved 5 chunks, and the answer is in chunk #3
chunks_text = [
    "Chunk 1: Irrelevant information about vacation days...",
    "Chunk 2: More irrelevant content about health insurance...",
    "Chunk 3: Equipment reimbursement: up to £500 per fiscal year for home office setup.",
    "Chunk 4: Unrelated content about parking policies...",
    "Chunk 5: More irrelevant information...",
]

# Test 1: Answer in the middle (bad)
middle_prompt = f"""Context:
{chr(10).join(chunks_text)}

Question: {query}
Answer:"""

# Test 2: Answer at the beginning (better)
beginning_prompt = f"""Context:
{chunks_text[2]}
{chunks_text[0]}
{chunks_text[1]}
{chunks_text[3]}
{chunks_text[4]}

Question: {query}
Answer:"""

print("When answer is buried in middle:")
print(llm.invoke(middle_prompt).content)
print("\nWhen answer is at beginning:")
print(llm.invoke(beginning_prompt).content)


# section4_compression.py
# You will also need a vector database and an llm
# to be defined, same as sections 2 and 3

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
