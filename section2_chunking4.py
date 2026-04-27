# section2_chunking4.py continued

# User query
query = "How do I process a refund for an item purchased 20 days ago?"

# Retrieve relevant chunks
# k=3 means "get the 3 most similar chunks"
relevant_docs = vectorstore.similarity_search(query, k=3)

print(f"Query: {query}\n")
print(f"Retrieved {len(relevant_docs)} chunks:\n")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Chunk {i}:")
    print(doc.page_content)
    print(f"(Length: {len(doc.page_content)} chars)")
    print("-" * 60)
