# test_legal_agent.py
"""
Quick test of the LegalAgent pipeline:
1. Loads FAISS index
2. Retrieves relevant evidence for a sample fact
3. Generates draft legal opinion using local LLM
"""

from src.legal_agent import LegalAgent

# Initialize LegalAgent (uses existing FAISS index and local LLM)
agent = LegalAgent(model_name='Model')  # Replace 'Model' with your local model folder if different

# Sample facts of a land-related case
sample_facts = """
The government issued a notice for land acquisition on the property
owned by Mr. Ramesh. The land is disputed, and Mr. Ramesh claims
that it falls under exempted agricultural lands. He seeks clarification
on applicable laws and relief options.
"""

print("\n=== Running retrieval + draft generation ===")
result = agent.run(facts=sample_facts, top_k=5)

# Show retrieved evidence
print("\n--- Top Retrieved Evidence ---")
for i, r in enumerate(result['retrieved']):
    print(f"[{i+1}] Source: {r['meta'].get('act_name','unknown')} | Score: {r['score']:.3f}")
    print(r['text'][:300] + '...\n')

# Show generated draft opinion
print("\n--- Draft Opinion ---")
print(result['opinion'])
