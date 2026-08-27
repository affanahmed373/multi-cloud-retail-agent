"""
Simple test script for the agent.

Run this to quickly test the agent with a few sample queries.
"""

from backend.app.agent import RetailAgent
from backend.app.llm_providers import MockLLMProvider
from backend.app.retriever import StoreRetriever


def main():
    # Initialize components
    retriever = StoreRetriever(
        products_path="data/products.json",
        policies_dir="data/policies",
    )
    llm = MockLLMProvider()
    agent = RetailAgent(llm, retriever=retriever)

    # Test queries
    queries = [
        "Do you ship to Berlin?",
        "What is your return policy?",
        "Do you have the Embroidered Black Shalwar Kameez in size M?",
        "I need something for a summer wedding under 80 EUR. What do you recommend?",
        "Which items are low in stock?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("="*60)
        result = agent.handle_query(q)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()