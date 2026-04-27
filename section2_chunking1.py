# section2_chunking.py continued
from config.llm_config import get_embeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

embeddings = get_embeddings()

sample_documents = [
    """TechRetail Employee Handbook - Section 4.7: Remote Work Policy
All Level 3+ employees are entitled to 15 remote work days per quarter.
Remote days must be requested 48 hours in advance via the HR portal.
Managers may approve additional days for exceptional circumstances.
Equipment reimbursement: up to £500 per fiscal year for home office setup.""",

    """TechRetail Employee Handbook - Section 5.2: Annual Leave Policy
All full-time employees receive 25 days of annual leave per year.
Leave must be approved by your line manager at least 2 weeks in advance.
Unused leave may be carried over up to a maximum of 5 days into the next year.
Bank holidays are in addition to the annual leave entitlement.""",

    """TechRetail Employee Handbook - Section 6.1: Performance Review Process
Performance reviews are conducted twice per year: in June and December.
Employees are assessed on objectives set at the start of the review period.
Ratings range from 1 (Below Expectations) to 5 (Exceptional).
A rating of 3 or above is required to be eligible for an annual pay review.""",
]

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

print("\n[OK] Vector database created and populated")
