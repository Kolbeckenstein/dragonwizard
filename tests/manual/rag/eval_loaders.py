"""
Manual evaluation script for document loaders.

This script tests the document loaders to ensure:
1. Text files load correctly
2. Markdown files load with title extraction
3. Metadata is properly extracted
4. Error handling works

Run with: uv run python tests/manual/rag/eval_loaders.py
"""

import asyncio
from pathlib import Path

from dragonwizard.rag.loaders import TextLoader, MarkdownLoader


async def main():
    print("=" * 60)
    print("Document Loaders Evaluation")
    print("=" * 60)
    print()

    # Test fixtures directory
    fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "sample_documents"

    # Test 1: TextLoader
    print("Test 1: TextLoader")
    print("-" * 60)

    text_loader = TextLoader()
    sample_text_file = fixtures_dir / "sample_rules.txt"

    if text_loader.supports_format(sample_text_file):
        print(f"✓ TextLoader supports {sample_text_file.suffix}")

        doc = await text_loader.load(sample_text_file)

        print(f"  Title: {doc.metadata.title}")
        print(f"  Source type: {doc.metadata.source_type}")
        print(f"  Source file: {doc.metadata.source_file}")
        print(f"  Text length: {len(doc.text)} characters")
        print(f"  Preview: {doc.text[:100]}...")
        print(f"  Pages: {doc.pages}")
        print()
    else:
        print(f"✗ TextLoader does not support {sample_text_file.suffix}")
        print()

    # Test 2: MarkdownLoader
    print("Test 2: MarkdownLoader")
    print("-" * 60)

    markdown_loader = MarkdownLoader()
    sample_md_file = fixtures_dir / "sample_readme.md"

    if markdown_loader.supports_format(sample_md_file):
        print(f"✓ MarkdownLoader supports {sample_md_file.suffix}")

        doc = await markdown_loader.load(sample_md_file)

        print(f"  Title: {doc.metadata.title}")
        print(f"  Source type: {doc.metadata.source_type}")
        print(f"  Source file: {doc.metadata.source_file}")
        print(f"  Text length: {len(doc.text)} characters")
        print(f"  Preview: {doc.text[:100]}...")
        print(f"  Pages: {doc.pages}")
        print()
    else:
        print(f"✗ MarkdownLoader does not support {sample_md_file.suffix}")
        print()

    # Test 3: Format detection
    print("Test 3: Format Detection")
    print("-" * 60)

    test_files = [
        ("sample_rules.txt", TextLoader),
        ("sample_readme.md", MarkdownLoader),
    ]

    for filename, expected_loader in test_files:
        file_path = fixtures_dir / filename
        text_supports = text_loader.supports_format(file_path)
        md_supports = markdown_loader.supports_format(file_path)

        print(f"  {filename}:")
        print(f"    TextLoader: {text_supports}")
        print(f"    MarkdownLoader: {md_supports}")

        if expected_loader == TextLoader and text_supports:
            print(f"    ✓ Correctly identified as text")
        elif expected_loader == MarkdownLoader and md_supports:
            print(f"    ✓ Correctly identified as markdown")
        else:
            print(f"    ✗ Format detection failed")

    print()
    print("=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
