"""
Tool functions for the retail agent.

This module provides:
- Inventory lookup.
- Policy document retrieval.
- Helper functions for product queries.

All functions work with local JSON/Markdown files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_products(products_path: str = "data/products.json") -> List[Dict[str, Any]]:
    """Load products from JSON file."""
    with open(products_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_inventory(
    sku: Optional[str] = None,
    size: Optional[str] = None,
    color: Optional[str] = None,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    occasion: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Check inventory based on filters.

    Returns a list of products matching the criteria.
    """
    products = load_products()
    results = []

    for p in products:
        # SKU filter
        if sku and p.get("sku") != sku:
            continue

        # Size filter
        if size and size not in p.get("sizes", []):
            continue

        # Color filter (partial match)
        if color:
            colors_lower = [c.lower() for c in p.get("colors", [])]
            if not any(color.lower() in c for c in colors_lower):
                continue

        # Category filter
        if category and p.get("category") != category:
            continue

        # Price filter
        if max_price is not None and p.get("price_eur", float("inf")) > max_price:
            continue

        # Occasion filter (partial match)
        if occasion:
            occasions_lower = [o.lower() for o in p.get("occasion", [])]
            if not any(occasion.lower() in o for o in occasions_lower):
                continue

        # Check stock > 0 for at least one size
        stock = p.get("stock", {})
        if all(qty == 0 for qty in stock.values()):
            continue

        results.append(p)

    return results


def get_policy(policy_id: str) -> str:
    """
    Get a policy document by ID.

    policy_id should match the filename without extension,
    e.g. "shipping_policy", "returns_policy".
    """
    policies_dir = Path("data/policies")
    policy_path = policies_dir / f"{policy_id}.md"

    if not policy_path.exists():
        return f"Policy '{policy_id}' not found."

    return policy_path.read_text(encoding="utf-8")


def get_product_by_sku(sku: str) -> Optional[Dict[str, Any]]:
    """Get a single product by SKU."""
    products = load_products()
    for p in products:
        if p.get("sku") == sku:
            return p
    return None


def list_low_stock(threshold: int = 3) -> List[Dict[str, Any]]:
    """
    List products with low stock.

    Returns products where any size has stock <= threshold.
    """
    products = load_products()
    low_stock_items = []

    for p in products:
        stock = p.get("stock", {})
        low_sizes = [size for size, qty in stock.items() if qty <= threshold]
        if low_sizes:
            item = p.copy()
            item["low_stock_sizes"] = low_sizes
            low_stock_items.append(item)

    return low_stock_items


# Simple manual test
if __name__ == "__main__":
    print("=== Inventory check: size M, color black ===")
    results = check_inventory(size="M", color="black")
    for r in results:
        print(f"- {r['name']} ({r['sku']}): {r['stock']}")

    print("\n=== Low stock items ===")
    low = list_low_stock(threshold=3)
    for item in low:
        print(f"- {item['name']} ({item['sku']}): low in {item['low_stock_sizes']}")

    print("\n=== Returns policy ===")
    policy = get_policy("returns_policy")
    print(policy[:300], "...")