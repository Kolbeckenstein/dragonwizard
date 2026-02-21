"""
Manual evaluation script for SentenceAwareChunker.

This script tests the chunking strategy to ensure:
1. Text is split at sentence boundaries
2. Chunks respect token limits
3. Overlapping works correctly
4. Metadata is generated properly

Run with: uv run python tests/manual/rag/eval_chunking.py
"""

from dragonwizard.rag.chunking import SentenceAwareChunker


def main():
    print("=" * 60)
    print("Chunking Strategy Evaluation")
    print("=" * 60)
    print()

    # Sample D&D text with multiple sentences
    text = """
    Fireball is a 3rd-level evocation spell. When you cast this spell, you hurl a tiny mote of fire at a point within range. The mote explodes in a 20-foot-radius sphere of fire. Each creature in that area must make a Dexterity saving throw. A target takes 8d6 fire damage on a failed save, or half as much damage on a successful one.

    The fire spreads around corners. It ignites flammable objects in the area that aren't being worn or carried.

    At Higher Levels: When you cast this spell using a spell slot of 4th level or higher, the damage increases by 1d6 for each slot level above 3rd.

    Magic Missile is a 1st-level evocation spell. You create three glowing darts of magical force. Each dart hits a creature of your choice that you can see within range. A dart deals 1d4 + 1 force damage to its target. The darts all strike simultaneously, and you can direct them to hit one creature or several.

    At Higher Levels: When you cast this spell using a spell slot of 2nd level or higher, the spell creates one more dart for each slot level above 1st.
    """

    print("Test text:")
    print(f"  Length: {len(text)} characters")
    print(f"  Preview: {text.strip()[:100]}...")
    print()

    # Initialize chunker with small target for testing
    chunker = SentenceAwareChunker(
        target_tokens=80,  # Small target to force multiple chunks
        overlap_tokens=15
    )

    print(f"Chunker configuration:")
    print(f"  Target tokens: {chunker.target_tokens}")
    print(f"  Overlap tokens: {chunker.overlap_tokens}")
    print(f"  Encoding: {chunker.encoding_name}")
    print()

    # Chunk the text
    metadata = {
        "source_file": "srd.pdf",
        "source_type": "pdf",
        "title": "D&D 5e SRD",
        "page_number": 241
    }

    chunks = chunker.chunk_text(
        text=text,
        document_id="doc-test-123",
        metadata=metadata
    )

    print(f"✓ Created {len(chunks)} chunks")
    print()

    # Analyze chunks
    print("Chunk analysis:")
    print()

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}/{len(chunks)}:")
        print(f"  ID: {chunk.metadata.chunk_id}")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print(f"  Page: {chunk.metadata.page_number}")
        print(f"  Text preview: {chunk.text[:80]}...")
        print()

    # Verify constraints
    print("Validation:")

    # Check token limits
    max_tokens = max(c.metadata.token_count for c in chunks)
    if max_tokens <= chunker.target_tokens * 1.2:  # Allow 20% tolerance
        print(f"  ✓ All chunks within token limit (max: {max_tokens})")
    else:
        print(f"  ✗ Some chunks exceed limit (max: {max_tokens})")

    # Check overlap (adjacent chunks should share some text)
    if len(chunks) > 1:
        chunk1_words = set(chunks[0].text.split())
        chunk2_words = set(chunks[1].text.split())
        overlap_words = chunk1_words & chunk2_words

        if overlap_words:
            print(f"  ✓ Chunks have overlap ({len(overlap_words)} shared words)")
        else:
            print(f"  ✗ No overlap detected between chunks")

    # Check metadata
    if all(c.metadata.document_id == "doc-test-123" for c in chunks):
        print(f"  ✓ All chunks have correct document_id")
    else:
        print(f"  ✗ Document IDs inconsistent")

    if all(c.metadata.source_file == "srd.pdf" for c in chunks):
        print(f"  ✓ All chunks have correct source_file")
    else:
        print(f"  ✗ Source files inconsistent")

    # Check chunk indices
    expected_indices = list(range(len(chunks)))
    actual_indices = [c.metadata.chunk_index for c in chunks]
    if actual_indices == expected_indices:
        print(f"  ✓ Chunk indices are sequential")
    else:
        print(f"  ✗ Chunk indices incorrect: {actual_indices}")

    # Check total_chunks
    if all(c.metadata.total_chunks == len(chunks) for c in chunks):
        print(f"  ✓ All chunks have correct total_chunks value")
    else:
        print(f"  ✗ total_chunks values inconsistent")

    print()
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
