# section1_knowledge_boundary.py
from config.llm_config import get_llm

llm = get_llm(temperature=0)

# Our fictional company policy (not in any LLM's training data)
secret_policy = """
TechRetail Employee Handbook - Confidential
Section 4.7: Remote Work Policy

All Level 3+ employees are entitled to 15 remote work days per quarter.
Remote days must be requested 48 hours in advance via the HR portal.
Managers may approve additional days for exceptional circumstances.
Equipment reimbursement: up to £500 per fiscal year for home office setup.
"""

# Try asking the LLM about this policy WITHOUT giving it the context
query = "What is TechRetail's remote work policy for Level 3 employees?"

response = llm.invoke(query)
print("Without RAG:")
print(response.content)
print("\n" + "="*60 + "\n")

missing_query = "What is TechRetail's parental leave policy?"
missing_prompt = f"""Use the following company policy document to answer the question.

Document:
{secret_policy}

Question: {missing_query}

Answer based only on the document above. If the information isn't present, say so clearly:"""

response = llm.invoke(missing_prompt)
print("\nMissing information test:")
print(response.content)
