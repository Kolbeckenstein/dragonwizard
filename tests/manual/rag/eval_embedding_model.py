"""
Manual evaluation script for EmbeddingModel.

This script tests the embedding model to ensure:
1. It loads successfully
2. It generates embeddings of the correct dimension
3. Similar texts produce similar embeddings
4. Different texts produce different embeddings

Run with: uv run python tests/manual/rag/eval_embedding_model.py
"""

import asyncio

import numpy as np

from dragonwizard.rag.embeddings import EmbeddingModel


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b))  # Already normalized, so dot product = cosine sim


async def main():
    print("=" * 60)
    print("Embedding Model Evaluation")
    print("=" * 60)
    print()

    # Test texts
    texts = [
        "Fireball deals 8d6 fire damage to creatures in a 20-foot radius",
        "Cone of Cold deals 8d8 cold damage in a 60-foot cone",
        "The wizard casts a spell that deals damage",
        "How do I bake a chocolate cake?",
    ]

    print("Test texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    print()

    # Initialize model
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    async with EmbeddingModel(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=4
    ) as model:
        print("✓ Model loaded successfully")
        print()

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = await model.embed(texts)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        print(f"  - Number of texts: {embeddings.shape[0]}")
        print(f"  - Embedding dimension: {embeddings.shape[1]}")
        print()

        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"Embedding norms (should all be ~1.0):")
        for i, norm in enumerate(norms, 1):
            print(f"  Text {i}: {norm:.6f}")
        print()

        # Compute similarity matrix
        print("Similarity matrix (cosine similarity):")
        print("     ", end="")
        for i in range(len(texts)):
            print(f"  T{i+1}  ", end="")
        print()

        for i in range(len(texts)):
            print(f"  T{i+1}", end="")
            for j in range(len(texts)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                print(f" {sim:5.3f}", end="")
            print()
        print()

        # Analyze semantic relationships
        print("Semantic analysis:")
        print(f"  Similarity(Fireball, Cone of Cold): {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
        print("    → Both are D&D damage spells (should be high)")
        print()
        print(f"  Similarity(Fireball, wizard casts): {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
        print("    → Related to spellcasting (should be moderate)")
        print()
        print(f"  Similarity(Fireball, chocolate cake): {cosine_similarity(embeddings[0], embeddings[3]):.3f}")
        print("    → Completely unrelated (should be low)")
        print()

        # Expected outcomes
        spell_sim = cosine_similarity(embeddings[0], embeddings[1])
        unrelated_sim = cosine_similarity(embeddings[0], embeddings[3])

        print("Validation:")
        if spell_sim > 0.5:
            print("  ✓ Similar spells have high similarity")
        else:
            print(f"  ✗ Similar spells should be more similar (got {spell_sim:.3f})")

        if unrelated_sim < 0.3:
            print("  ✓ Unrelated texts have low similarity")
        else:
            print(f"  ✗ Unrelated texts should be less similar (got {unrelated_sim:.3f})")

        if spell_sim > unrelated_sim:
            print("  ✓ Semantic relationships are captured correctly")
        else:
            print("  ✗ Semantic relationships not captured properly")

    print()
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
