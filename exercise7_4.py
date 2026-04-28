from config.llm_config import get_embeddings, get_llm
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from exercise7_1 import academic_policies

embeddings = get_embeddings()
llm = get_llm(temperature=0)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_text("\n\n".join(academic_policies))

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="acad_policy_ex74",
    ids=[str(i) for i in range(len(chunks))]
)

# ── Test set: 10 query-answer pairs ──────────────────────────────────────────
# Each entry has a query, expected answer keyword, and difficulty level.
test_set = [
    {
        "query": "What minimum attendance percentage is required to sit exams?",
        "expected": "80%",
        "difficulty": "easy",
    },
    {
        "query": "What is the late submission penalty per day?",
        "expected": "10% of the available mark per day",
        "difficulty": "easy",
    },
    {
        "query": "How many working days to submit a grade appeal?",
        "expected": "10 working days",
        "difficulty": "easy",
    },
    {
        "query": "How early must students arrive before an exam?",
        "expected": "at least 15 minutes",
        "difficulty": "easy",
    },
    {
        "query": "How soon must medical documentation be provided after returning?",
        "expected": "within 48 hours",
        "difficulty": "medium",
    },
    {
        "query": "For how many calendar days is late submission accepted?",
        "expected": "5 calendar days",
        "difficulty": "medium",
    },
    {
        "query": "What grade does a student receive after the 5-day late window?",
        "expected": "zero",
        "difficulty": "medium",
    },
    {
        "query": "Is self-plagiarism a violation under the academic integrity policy?",
        "expected": "yes, reusing own work without disclosure may be a violation",
        "difficulty": "medium",
    },
    {
        "query": "What are the possible consequences of repeated poor attendance?",
        "expected": "academic warning, loss of assessment eligibility, module failure",
        "difficulty": "hard",
    },
    {
        "query": "Who is responsible for maintaining accurate attendance records?",
        "expected": "module coordinators",
        "difficulty": "hard",
    },
]

# ── Helper functions ──────────────────────────────────────────────────────────

def build_rag_prompt(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return (
        "You are a university academic advisor. Answer based only on the "
        "policy documentation below.\n\n"
        f"Policy Documentation:\n{context}\n\n"
        f"Question: {query}\n\n"
        "If the answer is not in the documentation, say so clearly.\nAnswer:"
    )

def score_accuracy(response, expected):
    prompt = (
        f"Expected key information: \"{expected}\"\n"
        f"Response: \"{response[:300]}\"\n\n"
        "Rate accuracy 0, 1, or 2:\n"
        "2 = response clearly contains the expected information\n"
        "1 = response is partially correct or too vague\n"
        "0 = response is wrong or admits it does not know\n\n"
        "Reply with a single digit only."
    )
    result = llm.invoke(prompt)
    try:
        return max(0, min(2, int(result.content.strip()[0])))
    except (ValueError, IndexError):
        return 1

def check_faithfulness(response_text, context_text):
    prompt = (
        f"Context (first 500 chars): {context_text[:500]}\n\n"
        f"Response: {response_text[:300]}\n\n"
        "Does the response contradict any information in the context?\n"
        "Reply with YES or NO and one sentence of explanation."
    )
    return llm.invoke(prompt).content.strip()

# ── Run evaluation ────────────────────────────────────────────────────────────

print("#"*60)
print("# Exercise 7.4: Comprehensive Evaluation Framework")
print("#"*60)
print(f"Test set: {len(test_set)} queries | Vectorstore: {len(chunks)} chunks\n")

results = []

for entry in test_set:
    query      = entry["query"]
    expected   = entry["expected"]
    difficulty = entry["difficulty"]

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    response = llm.invoke(build_rag_prompt(query, docs))
    answer = response.content.strip()

    accuracy    = score_accuracy(answer, expected)
    faith_check = check_faithfulness(answer, context)
    faithful    = "YES" not in faith_check.upper()

    results.append({
        "query":      query[:48],
        "difficulty": difficulty,
        "accuracy":   accuracy,
        "faithful":   faithful,
    })

    print(f"Q: {query}")
    preview = answer[:150].replace("\n", " ")
    print(f"A: {preview}...")
    print(f"Expected  : {expected}")
    print(f"Accuracy  : {accuracy}/2 | Faithfulness: {faith_check[:55]}")
    print(f"Difficulty: {difficulty}")
    print("-"*60)

# ── Aggregate metrics ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("AGGREGATE RESULTS")
print("="*60)

total_acc  = sum(r["accuracy"] for r in results)
total_max  = len(results) * 2
faith_pass = sum(1 for r in results if r["faithful"])

print(f"Total accuracy  : {total_acc}/{total_max} ({total_acc/total_max*100:.0f}%)")
print(f"Average score   : {total_acc/len(results):.2f}/2.0")
print(f"Faithfulness    : {faith_pass}/{len(results)} responses non-contradictory")

for label, diff in [("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]:
    group = [r for r in results if r["difficulty"] == diff]
    if group:
        avg = sum(r["accuracy"] for r in group) / len(group)
        print(f"{label} queries ({len(group)}): avg {avg:.2f}/2.0")

print("\n" + "-"*75)
print(f"{'Query':<50} {'Diff':<8} {'Score':<7} Faithful")
print("-"*75)
for r in results:
    faith_str = "[OK]  " if r["faithful"] else "[WARN]"
    print(f"{r['query']:<50} {r['difficulty']:<8} {r['accuracy']}/2    {faith_str}")
