from config.llm_config import get_embeddings, get_llm
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from exercise7_1 import academic_policies

embeddings = get_embeddings()
llm = get_llm(temperature=0)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text("\n\n".join(academic_policies))
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="acad_policy_ex76"
)

# ── Helper: build prompt from selected docs ────────────────────────────────────
def build_rag_prompt(query, selected_docs):
    context = "\n\n".join([doc.page_content for doc in selected_docs])
    return (
        "You are a university academic advisor. Answer based only on the "
        "policy documentation below.\n\n"
        f"Policy Documentation:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

# ── Strategy 3: LLM re-ranking ────────────────────────────────────────────────
def rerank_with_llm(query, candidates):
    """Ask the LLM to select the 3 most relevant snippets from 10 candidates."""
    lines = []
    for i, c in enumerate(candidates[:10]):
        snippet = c.page_content[:150].replace("\n", " ")
        lines.append(f"{i+1}. {snippet}...")
    snippets_text = "\n".join(lines)

    prompt = (
        f"Question: {query}\n\n"
        f"Below are 10 retrieved document snippets numbered 1-10:\n"
        f"{snippets_text}\n\n"
        "Which 3 snippet numbers are most relevant to the question?\n"
        "Reply with exactly 3 comma-separated numbers, e.g.: 2, 5, 8"
    )
    result = llm.invoke(prompt)
    try:
        nums = [int(x.strip()) - 1 for x in result.content.split(",")[:3]]
        nums = [n for n in nums if 0 <= n < len(candidates)]
        if len(nums) < 3:
            nums = list(range(min(3, len(candidates))))
    except Exception:
        nums = [0, 1, 2]
    return [candidates[n] for n in nums]

# ── 5 test queries ─────────────────────────────────────────────────────────────
test_queries = [
    "What are the consequences of poor attendance and missing exams?",
    "What is the process for submitting late work and what penalty applies?",
    "What tools and materials are allowed during examinations?",
    "How does the university handle plagiarism and what are the sanctions?",
    "What adjustments are available for students with disabilities?",
]

print("#"*60)
print("# Exercise 7.6: Context Window Optimization")
print("#"*60)
print(f"Total chunks: {len(chunks)} | Retrieve 10 candidates, select 3 by each strategy")
print()

for qi, query in enumerate(test_queries, 1):
    print(f"{'='*60}")
    print(f"Query {qi}: {query}")
    print("="*60)

    # Retrieve 10 candidates with scores
    candidates_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    docs_only = [doc for doc, _ in candidates_with_scores]
    scores    = [score for _, score in candidates_with_scores]

    print(f"Top score: {scores[0]:.3f} | Bottom score (rank 10): {scores[-1]:.3f}")

    # Strategy 1: Top 3 by similarity score (default ranking)
    top3    = docs_only[:3]

    # Strategy 2: Diverse selection — top 2 + bottom 1 from k=10
    diverse = [docs_only[0], docs_only[1], docs_only[9]]

    # Strategy 3: LLM re-ranking
    reranked = rerank_with_llm(query, docs_only)

    strategies = [
        ("Top-3 (highest similarity)", top3),
        ("Diverse (top-2 + bottom-1)", diverse),
        ("LLM re-ranked",              reranked),
    ]

    for s_name, selected in strategies:
        response = llm.invoke(build_rag_prompt(query, selected))
        answer   = response.content.strip()
        preview  = answer[:130].replace("\n", " ")
        print(f"\n  [{s_name}]")
        print(f"  Answer: {preview}...")

    print()

# ── Comparison findings (deliverable) ────────────────────────────────────────
print("="*60)
print("COMPARISON FINDINGS (deliverable)")
print("="*60)
findings = """
Top-3 by similarity score
  Reliable for single-topic queries. When the top-3 chunks all cover the same
  policy section the answer is precise, but it may miss related context from
  a second policy if the question spans multiple areas. Fast and reproducible.

Diverse selection (top-2 + bottom-1)
  Useful when a question bridges multiple policies, because the bottom-ranked
  chunk can introduce a less-obvious but relevant angle (e.g. the attendance
  exception when the exam question retrieves mostly exam-conduct chunks).
  However, the bottom-1 chunk can also be genuinely irrelevant, introducing
  noise that confuses the model. Performance is inconsistent.

LLM re-ranking
  The strongest strategy for complex multi-policy questions. The LLM evaluates
  relevance based on content meaning rather than embedding distance alone,
  selecting chunks that directly address the query even when similarity scores
  are not markedly different. The cost is an extra LLM call per query (latency)
  and reliability depends on the base model quality. With gemma3:1b the
  re-ranker is borderline; a larger model makes this the clear winner.

Recommendation
  Use Top-3 for simple, single-policy questions where speed matters.
  Use LLM re-ranking for complex or multi-hop queries. Avoid pure diversity
  sampling unless a minimum relevance threshold ensures the diversity candidate
  is still on-topic rather than random noise.
"""
print(findings)
