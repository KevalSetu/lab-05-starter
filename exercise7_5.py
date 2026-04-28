from config.llm_config import get_embeddings, get_llm
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from exercise7_1 import academic_policies

embeddings = get_embeddings()
llm = get_llm(temperature=0)

all_text   = "\n\n".join(academic_policies)
test_query = "What is the minimum attendance required to pass a module?"
expected   = "80%"

def prompt_basic(context, query):
    return f"Context: {context}\nQuestion: {query}\nAnswer:"

def prompt_no_instructions(context, query):
    return f"{context}\n\n{query}"

def run_failure_test(label, vectorstore, query, prompt_fn, k=3):
    docs    = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    answer  = llm.invoke(prompt_fn(context, query)).content.strip()
    preview = answer[:180].replace("\n", " ")
    print(f"Response: {preview}...")
    return answer

print("#"*60)
print("# Exercise 7.5: Failure Mode Analysis")
print("#"*60)
print(f"Baseline query: \"{test_query}\"")
print(f"Expected answer contains: \"{expected}\"\n")

# ─────────────────────────────────────────────────────────────────────────────
# First build the normal vectorstore (used by modes 2-4)
normal_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
normal_chunks   = normal_splitter.split_text(all_text)
vs_normal = Chroma.from_texts(
    texts=normal_chunks, embedding=embeddings, collection_name="ex75_normal"
)
# ─────────────────────────────────────────────────────────────────────────────

print("="*60)
print("FAILURE MODE 1: Terrible chunking (chunk_size=50, no overlap)")
print("="*60)
print("Change: RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)")

tiny_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
tiny_chunks   = tiny_splitter.split_text(all_text)
vs_tiny = Chroma.from_texts(
    texts=tiny_chunks, embedding=embeddings, collection_name="ex75_tiny"
)
print(f"Chunks created: {len(tiny_chunks)} (avg {sum(len(c) for c in tiny_chunks)/len(tiny_chunks):.0f} chars)")
print(f"Example chunk : '{tiny_chunks[10]}'")
run_failure_test("Tiny chunks", vs_tiny, test_query, prompt_basic)
print("Root cause : 50-char chunks break sentences mid-phrase. Retrieved fragments")
print("             lack enough context for the model to infer any policy rule.")
print("Fix        : Use chunk_size >= 300 and preserve sentence boundaries.")

print()
print("="*60)
print("FAILURE MODE 2: Too few chunks retrieved (k=1)")
print("="*60)
print("Change: vectorstore.similarity_search(query, k=1) instead of k=3")
run_failure_test("k=1", vs_normal, test_query, prompt_basic, k=1)
print("Root cause : The 80% attendance rule and its exceptions are spread across")
print("             multiple sections. k=1 returns only one section; the model")
print("             may miss the exception clauses or the exact threshold.")
print("Fix        : Use k=3-5; validate that retrieved chunks cover the full rule.")

print()
print("="*60)
print("FAILURE MODE 3: Too many chunks retrieved (k=20)")
print("="*60)
print("Change: vectorstore.similarity_search(query, k=20) floods context")
run_failure_test("k=20", vs_normal, test_query, prompt_basic, k=20)
print("Root cause : With 20 chunks, most are from unrelated policies (exam rules,")
print("             integrity policy). The relevant rule is buried in the middle,")
print("             token count spikes, and the model blends rules from policies.")
print("Fix        : Cap k at 3-5; use contextual compression to trim noise.")

print()
print("="*60)
print("FAILURE MODE 4: Prompt with no instructions")
print("="*60)
print("Change: Raw context dump followed by query, no role or constraints")
run_failure_test("No instructions", vs_normal, test_query, prompt_no_instructions)
print("Root cause : Without a role or cite-only instruction the model may answer")
print("             from its general training knowledge rather than the context,")
print("             or pad the response with unnecessary caveats.")
print("Fix        : Always include role, cite-only constraint, and fallback directive.")

print()
print("="*60)
print("FAILURE MODE 5: Out-of-domain query")
print("="*60)
ood_query = "What is the university's parking permit application process?"
print(f"Change: query has no matching content in the knowledge base")
print(f"Query : {ood_query}")
docs    = vs_normal.similarity_search(ood_query, k=3)
context = "\n\n".join([d.page_content for d in docs])
answer  = llm.invoke(prompt_basic(context, ood_query)).content.strip()
preview = answer[:200].replace("\n", " ")
print(f"Response: {preview}...")

admits = any(p in answer.lower() for p in
             ["not", "no information", "unable", "don't", "cannot", "does not"])
if admits:
    print("[OK]   Model correctly admitted it lacks relevant information")
else:
    print("[WARN] Model may have hallucinated an answer not in the policies")

print("Root cause : Retrieval returns the least-bad match (typically exam/attendance")
print("             policy) which is irrelevant. Without a relevance threshold,")
print("             the model answers from the irrelevant chunks or prior knowledge.")
print("Fix        : Add a similarity score threshold (reject if top score < 0.4);")
print("             include instruction 'If no relevant policy exists, say so.'")

print()
print("="*60)
print("FAILURE MODE SUMMARY")
print("="*60)
summary = [
    ("Tiny chunks (50 chars)",         "Fragmented context, incoherent answers",
     "chunk_size >= 300, preserve sentences"),
    ("k=1 retrieval",                  "Misses related sections / exceptions",
     "k=3-5, validate coverage"),
    ("k=20 retrieval",                 "Noise overwhelms relevant content",
     "k=3-5, add compression"),
    ("No-instruction prompt",          "Model ignores context or adds caveats",
     "Role + cite-only + fallback directive"),
    ("Out-of-domain query",            "Hallucination or irrelevant retrieval",
     "Score threshold + explicit out-of-scope instruction"),
]
print(f"{'Mode':<28} {'Failure':<40} {'Fix'}")
print("-"*100)
for mode, failure, fix in summary:
    print(f"{mode:<28} {failure:<40} {fix}")
