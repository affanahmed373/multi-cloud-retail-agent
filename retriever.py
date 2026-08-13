"""
Simple BM25-based retriever for the clothing store.

This module:
- Loads products and policy documents.
- Builds a BM25 index over text chunks.
- Retrieves top-k chunks for a given query.

No cloud dependencies; runs fully locally.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


def load_products(products_path: str = "data/products.json") -> List[Dict[str, Any]]:
    with open(products_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_policies(policies_dir: str = "data/policies") -> List[Dict[str, str]]:
    """
    Load all .md files from policies_dir.
    Returns a list of docs with keys: id, text.
    """
    policies = []
    policies_path = Path(policies_dir)
    if not policies_path.exists():
        return policies

    for md_file in sorted(policies_path.glob("*.md")):
        doc_id = md_file.stem  # e.g. "shipping_policy"
        text = md_file.read_text(encoding="utf-8")
        policies.append({"id": doc_id, "text": text})

    return policies


def build_chunks(
    products: List[Dict[str, Any]],
    policies: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Build chunks for indexing.

    For products: one chunk per product (description + key fields).
    For policies: one chunk per policy document.

    Returns a list of dicts with keys: id, source_type, text.
    """
    chunks = []

    # Product chunks
    for p in products:
        text_parts = [
            f"SKU: {p['sku']}",
            f"Name: {p['name']}",
            f"Category: {p['category']}",
            f"Fabric: {p['fabric']}",
            f"Colors: {', '.join(p['colors'])}",
            f"Sizes: {', '.join(p['sizes'])}",
            f"Price: {p['price_eur']} EUR",
            f"Description: {p['short_description']}",
            f"Occasion: {', '.join(p['occasion'])}",
            f"Care: {p['care_instructions']}",
        ]
        # Add stock info as text
        stock_parts = [f"{size}: {qty}" for size, qty in p["stock"].items()]
        text_parts.append(f"Stock: {', '.join(stock_parts)}")

        text = ". ".join(text_parts)

        chunks.append({
            "id": p["sku"],
            "source_type": "product",
            "text": text,
        })

    # Policy chunks
    for pol in policies:
        chunks.append({
            "id": pol["id"],
            "source_type": "policy",
            "text": pol["text"],
        })

    return chunks


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer: lowercase + split on non-alphanumeric.
    Good enough for a small demo; can be improved later.
    """
    import re

    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class StoreRetriever:
    """
    BM25 retriever over products and policies.
    """

    def __init__(
        self,
        products_path: str = "data/products.json",
        policies_dir: str = "data/policies",
    ):
        self.products = load_products(products_path)
        self.policies = load_policies(policies_dir)
        self.chunks = build_chunks(self.products, self.policies)

        # Prepare BM25 index
        self.corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for a given query.

        Returns a list of dicts with keys: id, source_type, text, score.
        """
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        # Get top_k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results


# Simple manual test
if __name__ == "__main__":
    retriever = StoreRetriever()

    test_queries = [
        "Do you ship to Berlin?",
        "What is your return policy?",
        "Do you have black shalwar kameez in size M?",
        "I need something for a summer wedding under 80 EUR.",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"- [{r['source_type']}] {r['id']} (score={r['score']:.3f})")
            print(f"  {r['text'][:150]}...")