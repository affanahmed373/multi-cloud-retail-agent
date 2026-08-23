"""
Qdrant-based semantic retriever for the clothing store.

This module:
- Loads products and policy documents.
- Builds text chunks for indexing.
- Encodes chunks with all-MiniLM-L6-v2 embeddings.
- Stores embeddings in Qdrant and retrieves top-k chunks for a query.

Requires:
- qdrant-client
- sentence-transformers
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Deterministic UUIDs from string chunk ids (Qdrant only accepts UUID or unsigned int)
_POINT_ID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NS, chunk_id))

def load_products(products_path: str = "data/products.json") -> List[Dict[str, Any]]:
    with open(products_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_policy_chunks(policies: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    chunks = []
    for pol in policies:
        for i, chunk in enumerate(splitter.split_text(pol["text"])):
            chunks.append({
                "id": f"{pol['id']}_chunk_{i}",
                "source_type": "policy",
                "text": chunk,
                "payload": {
                    "policy_id": pol["id"],
                    "source_type": "policy",
                    "chunk_index": i,
                },
            })
    return chunks

def load_policies(policies_dir: str = "data/policies") -> List[Dict[str, str]]:
    """
    Load all .md files from policies_dir.
    Returns a list of docs with keys: id, text.
    """
    policies: List[Dict[str, str]] = []
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
) -> List[Dict[str, Any]]:
    """
    Build chunks for indexing.

    For products: one chunk per product (description + key fields).
    For policies: one chunk per policy document.

    Returns a list of dicts with keys:
    - id: unique identifier (SKU or policy id)
    - source_type: "product" or "policy"
    - text: chunk text
    - payload: metadata for filtering (category, price, etc.)
    """
    chunks: List[Dict[str, Any]] = []

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

        payload = {
            "sku": p["sku"],
            "name": p["name"],
            "category": p["category"],
            "fabric": p["fabric"],
            "colors": p["colors"],
            "sizes": p["sizes"],
            "price_eur": p["price_eur"],
            "occasion": p["occasion"],
            "source_type": "product",
        }

        chunks.append(
            {
                "id": p["sku"],
                "source_type": "product",
                "text": text,
                "payload": payload,
            }
        )

    # Policy chunks
    policy_chunks = build_policy_chunks(policies)
    chunks.extend(policy_chunks)

    return chunks


class StoreRetriever:
    """
    Qdrant-based semantic retriever over products and policies.

    Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings and
    Qdrant for vector search.
    """

    def __init__(
        self,
        products_path: str = "data/products.json",
        policies_dir: str = "data/policies",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "retail_store",
    ):
        self.products = load_products(products_path)
        self.policies = load_policies(policies_dir)
        self.chunks = build_chunks(self.products, self.policies)

        # Initialize encoder model
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize Qdrant client
        # qdrant_url can be your cloud URL or "http://localhost:6333" for local
        self.client = QdrantClient(
            url=qdrant_url or "http://localhost:6333",
            api_key=qdrant_api_key,
        )
        self.collection_name = collection_name

        # Ensure collection exists, then index data
        self._ensure_collection()
        self._index_chunks()

    def _ensure_collection(self) -> None:
        """
        Create collection if it does not exist.
        """
        if self.client.collection_exists(self.collection_name):
            return

        vector_dim = self.encoder.get_embedding_dimension()
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )
        except UnexpectedResponse as e:
            # Race / reload: another process created it first
            if "already exists" not in str(e).lower():
                raise

    def _index_chunks(self) -> None:
        """
        Encode all chunks and upsert them into Qdrant.
        """
        if not self.chunks:
            return

        texts = [c["text"] for c in self.chunks]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)

        points = []
        for chunk, emb in zip(self.chunks, embeddings):
            payload = {**chunk["payload"], "chunk_id": chunk["id"], "text": chunk["text"]}
            points.append(
                models.PointStruct(
                    id=_point_id(chunk["id"]),
                    vector=emb.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[models.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top_k chunks for a given query using semantic search.

        Optionally apply Qdrant payload filters (e.g., category, size).
        Returns a list of dicts with keys: id, source_type, text, score, payload.
        """
        query_vec = self.encoder.encode(query, normalize_embeddings=True).tolist()

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            query_filter=filters,
        ).points

        results: List[Dict[str, Any]] = []
        for point in search_result:
            payload = point.payload or {}
            source_type = payload.get("source_type", "unknown")
            chunk_id = payload.get("chunk_id", str(point.id))
            text = payload.get("text") or ""
            if not text:
                matching = next(
                    (c for c in self.chunks if c["id"] == chunk_id), None
                )
                text = matching["text"] if matching else ""

            results.append(
                {
                    "id": chunk_id,
                    "source_type": source_type,
                    "text": text,
                    "score": float(point.score),
                    "payload": payload,
                }
            )

        return results


# Simple manual test
if __name__ == "__main__":
    retriever = StoreRetriever(
        qdrant_url="http://localhost:6333",  # or your Qdrant Cloud URL
        qdrant_api_key=None,                # set if using Qdrant Cloud
    )

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