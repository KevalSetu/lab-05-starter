from config.llm_config import get_embeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = get_embeddings()

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

# Store chunks with IDs so we can track which ones are retrieved
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    ids=[str(i) for i in range(len(chunks))],
    collection_name="support_docs_retrieval_eval"
)

print(f"Indexed {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk[:80].strip()}...")

print()

def evaluate_retrieval(test_cases_with_labels, vectorstore, k=3):
    precision_scores = []
    recall_scores = []

    for test in test_cases_with_labels:
        query = test["query"]
        relevant_ids = set(str(i) for i in test["relevant_chunk_ids"])

        # Retrieve with IDs
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        retrieved_ids = set()
        collection = vectorstore._collection
        for doc, score in results:
            # Match chunk text back to its index
            for i, chunk in enumerate(chunks):
                if chunk == doc.page_content:
                    retrieved_ids.add(str(i))
                    break

        true_positives = len(retrieved_ids & relevant_ids)
        precision = true_positives / k if k > 0 else 0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Query: {query}")
        print(f"  Should retrieve chunks : {relevant_ids}")
        print(f"  Actually retrieved IDs : {retrieved_ids}")
        print(f"  Precision: {precision:.2f}  |  Recall: {recall:.2f}")
        print()

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall   : {avg_recall:.2f}")

    return precision_scores, recall_scores

test_with_labels = [
    {
        "query": "How do I process a refund?",
        "relevant_chunk_ids": [0, 1]
    },
    {
        "query": "What do I do about a lost package?",
        "relevant_chunk_ids": [2]
    },
]

evaluate_retrieval(test_with_labels, vectorstore)

print("Note: Full implementation requires tracking chunk IDs")
print("For your assignment, you'll build this properly")
