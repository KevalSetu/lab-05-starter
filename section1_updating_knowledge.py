# section1_updating_knowledge.py
from datetime import datetime

# Imagine this policy gets updated weekly
policy_v1 = "Remote work allowance: 15 days per quarter (valid until March 2025)"
policy_v2 = "Remote work allowance: 20 days per quarter (effective April 2025)"
policy_v3 = "Remote work allowance: Unlimited for L3+ (effective June 2025)"

# If we fine-tuned on policy_v1, we'd be stuck with outdated info
# RAG solves this: just update the document in your database

print("RAG advantage: Knowledge stays fresh without retraining")
print(f"Current policy (in our RAG database): {policy_v3}")
print("\nFine-tuning would require:")
print("- Collecting new training data")
print("- Retraining the entire model (expensive)")
print("- Redeploying the model")
print("- Potentially weeks of work")
print("\nRAG requires:")
print("- Updating one document in the database")
print("- Approximately 30 seconds")
