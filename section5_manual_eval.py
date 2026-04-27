# section5_manual_eval.py
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
    collection_name="support_docs_eval"
)

def build_rag_prompt(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return f"""You are a customer support assistant. Use the following support documentation to answer the question.

Documentation:
{context}

Question: {query}

Provide a clear, helpful answer based on the documentation. If the documentation doesn't contain enough information, say so.

Answer:"""

# Create evaluation test cases
test_cases = [
    {
        "query": "Can I return an item I bought on sale?",
        "expected_answer": "No, sale and clearance items are final sale",
        "category": "policy_knowledge"
    },
    {
        "query": "How long do I have to return a laptop?",
        "expected_answer": "14 days for electronic goods",
        "category": "policy_knowledge"
    },
    {
        "query": "What if the customer can't find their receipt?",
        "expected_answer": "Look up by email and order number, or escalate to supervisor",
        "category": "process_knowledge"
    },
]

# Run each test case through RAG
results = []

for test in test_cases:
    docs = vectorstore.similarity_search(test["query"], k=3)
    prompt = build_rag_prompt(test["query"], docs)
    response = llm.invoke(prompt)
    
    result = {
        "query": test["query"],
        "response": response.content,
        "expected": test["expected_answer"],
        "category": test["category"]
    }
    results.append(result)
    
    print(f"\nQuery: {test['query']}")
    print(f"Expected: {test['expected_answer']}")
    print(f"Got: {response.content}")
    print("-" * 60)

# Now manually score each response:
# 0 = wrong, 1 = partially correct, 2 = fully correct
# (In real evaluation, you'd save this to a file)


# section5_manual_eval.py continued

def check_faithfulness(query, retrieved_docs, response, llm):
    """Check if response is supported by retrieved context"""
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    check_prompt = f"""Query: {query}

Retrieved Context:
{context}

Generated Answer:
{response}

Is this answer fully supported by the retrieved context?
- YES if every claim in the answer comes from the context
- NO if the answer includes information not in the context
- PARTIAL if most claims are supported but some aren't

Provide your assessment (YES/NO/PARTIAL) and brief explanation:"""

    result = llm.invoke(check_prompt)
    return result.content

# Test it
query = "What's the refund policy for electronics?"
docs = vectorstore.similarity_search(query, k=3)
rag_response = llm.invoke(build_rag_prompt(query, docs))

faithfulness = check_faithfulness(query, docs, rag_response.content, llm)
print("Faithfulness check:")
print(faithfulness)


# section5_maunal_eval.py continued

# For this, you need a labelled test set where you know which chunks are relevant
# Here's a simplified version:

def evaluate_retrieval(test_cases_with_labels, vectorstore, k=3):
    """
    test_cases_with_labels format:
    [
        {
            "query": "...",
            "relevant_chunk_ids": [0, 2, 5]  # IDs of chunks that should be retrieved
        },
        ...
    ]
    """
    precision_scores = []
    recall_scores = []
    
    for test in test_cases_with_labels:
        query = test["query"]
        relevant_ids = set(test["relevant_chunk_ids"])
        
        # Retrieve
        retrieved = vectorstore.similarity_search(query, k=k)
        # (In real code, you'd track chunk IDs - this is simplified)
        
        # Calculate precision and recall
        # Precision: of what we retrieved, how many were relevant?
        # Recall: of what's relevant, how many did we retrieve?
        
        # (Simplified - you'd implement actual ID matching)
        print(f"Query: {query}")
        print(f"Should retrieve chunks: {relevant_ids}")
        print(f"Actually retrieved: {k} chunks")
    
    # This is a template - implement the actual metrics calculation
    return precision_scores, recall_scores

# Example structure:
test_with_labels = [
    {
        "query": "How do I process a refund?",
        "relevant_chunk_ids": [0, 1]  # Assuming we know chunks 0 and 1 contain the answer
    }
]

# evaluate_retrieval(test_with_labels, vectorstore)
print("Note: Full implementation requires tracking chunk IDs")
print("For your assignment, you'll build this properly")
