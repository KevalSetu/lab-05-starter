from config.llm_config import get_embeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from exercise7_1 import academic_policies

# Use the Attendance Policy as the long document (800+ words, rich structure)
long_document = academic_policies[0]

embeddings = get_embeddings()

print("#"*60)
print("# Exercise 7.2: Chunking Strategy Comparison")
print("#"*60)
print(f"Document length: {len(long_document)} characters")
print(f"Document: Attendance Policy\n")

# ── Strategy 1: Fixed-size (300 characters, no overlap) ──────────────────────
splitter_fixed = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=[" ", ""]
)
chunks_fixed = splitter_fixed.split_text(long_document)

# ── Strategy 2: Paragraph-based (large size, double-newline split first) ─────
splitter_para = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    separators=["\n\n", "\n"]
)
chunks_para = splitter_para.split_text(long_document)

# ── Strategy 3: Sentence groups with overlap ─────────────────────────────────
splitter_sent = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[". ", "\n", " "]
)
chunks_sent = splitter_sent.split_text(long_document)

# ── Report chunk statistics ───────────────────────────────────────────────────
strategies = [
    ("Fixed-size (300 chars, no overlap)", chunks_fixed),
    ("Paragraph-based (800 chars)",        chunks_para),
    ("Sentence groups (500 chars, 100 overlap)", chunks_sent),
]

for name, chunks in strategies:
    avg_len = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"Strategy: {name}")
    print(f"  Chunks created : {len(chunks)}")
    print(f"  Avg chunk length: {avg_len:.0f} chars")
    first_line = chunks[0][:100].strip().replace("\n", " ")
    print(f"  First chunk    : {first_line}...")
    print()

# ── Build vectorstores and compare retrieval ──────────────────────────────────
vs_fixed = Chroma.from_texts(texts=chunks_fixed, embedding=embeddings,
                              collection_name="ex72_fixed")
vs_para  = Chroma.from_texts(texts=chunks_para,  embedding=embeddings,
                              collection_name="ex72_para")
vs_sent  = Chroma.from_texts(texts=chunks_sent,  embedding=embeddings,
                              collection_name="ex72_sent")

test_queries = [
    "What is the minimum attendance percentage?",
    "What happens if a student has poor attendance?",
    "How is attendance recorded by the university?",
    "Are there exceptions to the attendance requirement?",
    "How should a student report a medical absence?",
]

print("#"*60)
print("# Retrieval comparison (k=1 best match per strategy)")
print("#"*60)

for query in test_queries:
    print(f"\nQuery: {query}")
    for label, vs in [("Fixed", vs_fixed), ("Paragraph", vs_para), ("Sentence", vs_sent)]:
        docs = vs.similarity_search(query, k=1)
        if docs:
            snippet = docs[0].page_content[:120].strip().replace("\n", " ")
        else:
            snippet = "No result"
        print(f"  [{label:10s}] {snippet}...")

# ── Written comparison (deliverable) ─────────────────────────────────────────
print("\n" + "="*60)
print("WRITTEN COMPARISON (deliverable)")
print("="*60)
comparison = """
Fixed-size (300 chars, no overlap)
  Produces the most chunks with the shortest average length. Chunks break at
  word boundaries but frequently cut mid-sentence, splitting related ideas
  across adjacent chunks. Retrieval returns fragments like "Students are
  normally expected to attend at least 80%" without the surrounding conditions
  that give it meaning. Coherence is poor for structured policy text.

Paragraph-based (800 chars, no overlap)
  Aligns splits to double-newline paragraph markers, keeping each numbered
  section (e.g. "3. General requirement") intact. Fewer, larger chunks mean
  each retrieved chunk delivers a complete, self-contained rule. Retrieval
  quality is noticeably better for direct policy questions.

Sentence groups with overlap (500 chars, 100 overlap)
  Produces a middle ground. The 100-char overlap ensures that sentences
  bridging two chunks appear in both, reducing the risk of a split cutting a
  key rule in half. Slightly more coherent than fixed-size, but numbered
  sections can still be split if a section is long.

Recommendation: Paragraph-based chunking for this document type.
The Attendance Policy uses numbered sections (1-10) where each section is a
self-contained, actionable rule. Splitting within a section breaks the logical
unit. Paragraph chunking preserves each rule intact, making retrieved chunks
directly usable in a prompt. The trade-off is fewer, larger chunks, but for
structured policy text coherence outweighs granularity. Word count: ~240.
"""
print(comparison)
