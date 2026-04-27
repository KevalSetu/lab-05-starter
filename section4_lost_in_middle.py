from config.llm_config import get_llm

llm = get_llm(temperature=0)

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
