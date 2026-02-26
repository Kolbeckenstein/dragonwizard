#!/usr/bin/env python3
"""
eval_pdf_strategies.py — Compare PDF ingestion strategy combinations side by side.

Runs (or re-uses) four named ChromaDB collections, each built with a different
combination of ExtractionMode and ChunkEnricher, then fires a battery of RAG
queries and prints a per-query comparison table so you can visually judge
retrieval quality across strategies.

Collections created
-------------------
  baseline      default extraction, no enrichment
  col_only      column-aware extraction, no enrichment
  stat_only     default extraction + statistical heading injection
  col_stat      column-aware + statistical heading injection
  weighted_only default extraction + weighted heading injection (score in prefix)
  col_weighted  column-aware + weighted heading injection   ← confidence-visible variant

Eval queries are chosen to surface two orthogonal failure modes:
  - Column interleaving:  stat block text that jumps between columns (beholder, vampire)
  - Missing section context:  chunks that lack a nearby heading anchor (orcs, fighter)

Usage
-----
  # Full run: ingest all PDFs then evaluate
  uv run python tests/manual/rag/eval_pdf_strategies.py

  # Skip ingestion if collections are already populated
  uv run python tests/manual/rag/eval_pdf_strategies.py --skip-ingest

  # Skip ingestion and retrieve more chunks
  uv run python tests/manual/rag/eval_pdf_strategies.py --skip-ingest --k 5

  # Ingest a single edition subdirectory instead of the full tree
  uv run python tests/manual/rag/eval_pdf_strategies.py --source data/raw/pdf/5e

  # Force re-ingest even if collections already have data
  uv run python tests/manual/rag/eval_pdf_strategies.py --force
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dragonwizard.config.settings import load_settings
from dragonwizard.rag.components import RAGComponents
from dragonwizard.rag.sources.pdf.loader import ExtractionMode
from dragonwizard.rag.sources.pdf.statistical_enricher import StatisticalHeadingEnricher
from dragonwizard.rag.sources.pdf.weighted_enricher import WeightedHeadingEnricher

# ---------------------------------------------------------------------------
# Strategy matrix
# ---------------------------------------------------------------------------

def _make_strategies() -> list[dict]:
    """Build strategy descriptors (enrichers are instantiated fresh each call)."""
    return [
        {
            "collection": "baseline",
            "extraction_mode": ExtractionMode.DEFAULT,
            "enrichers": [],
            "label": "baseline",
            "description": "default extraction, no enricher",
        },
        {
            "collection": "col_only",
            "extraction_mode": ExtractionMode.COLUMN_AWARE,
            "enrichers": [],
            "label": "col_only",
            "description": "column-aware extraction, no enricher",
        },
        {
            "collection": "stat_only",
            "extraction_mode": ExtractionMode.DEFAULT,
            "enrichers": [StatisticalHeadingEnricher()],
            "label": "stat_only",
            "description": "default extraction + statistical headings",
        },
        {
            "collection": "col_stat",
            "extraction_mode": ExtractionMode.COLUMN_AWARE,
            "enrichers": [StatisticalHeadingEnricher()],
            "label": "col_stat",
            "description": "column-aware + statistical headings",
        },
        {
            "collection": "weighted_only",
            "extraction_mode": ExtractionMode.DEFAULT,
            "enrichers": [WeightedHeadingEnricher()],
            "label": "weighted_only",
            "description": "default extraction + weighted headings (score in prefix)",
        },
        {
            "collection": "col_weighted",
            "extraction_mode": ExtractionMode.COLUMN_AWARE,
            "enrichers": [WeightedHeadingEnricher()],
            "label": "col_weighted  ★",
            "description": "column-aware + weighted headings (confidence-visible)",
        },
    ]


# ---------------------------------------------------------------------------
# Eval query suite
# ---------------------------------------------------------------------------

EVAL_QUERIES = [
    {
        "label": "Creature trait lookup",
        "query": "Do orcs have darkvision?",
        "edition": "5e",
        "note": "stat_only / col_stat should show [Section: Orc] prefix in top chunk",
    },
    {
        "label": "Stat block — AC and HP",
        "query": "What is the armor class and hit points of a beholder?",
        "edition": "5e",
        "note": "col_* should return coherent AC/HP in one chunk; baseline may interleave columns",
    },
    {
        "label": "Legendary actions (multi-column stat block)",
        "query": "What legendary actions can a vampire take?",
        "edition": "5e",
        "note": "col_* keeps Legendary Actions block in correct reading order",
    },
    {
        "label": "Class features (multi-column PHB table)",
        "query": "What class features does a fighter get at level 1?",
        "edition": "5e",
        "note": "col_* should return the feature table without left/right column interleaving",
    },
    {
        "label": "Spell rules — roughly equal across strategies",
        "query": "How does concentration work for spells?",
        "edition": None,
        "note": "All strategies should find the same concentration rules block; scores should be similar",
    },
    {
        "label": "2024 PHB — 5.5e edition filter",
        "query": "What changed for Paladins in the 2024 Player's Handbook?",
        "edition": "5.5e",
        "note": "Should return 5.5e content only; missing if PlayersHandbook2024.pdf not ingested",
    },
]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

async def ingest_strategy(strategy: dict, source_path: Path, settings, force: bool) -> None:
    """Ingest source_path into one collection using the given strategy."""
    collection = strategy["collection"]
    rag_settings = settings.rag.model_copy(update={"collection_name": collection})
    factory = RAGComponents(rag_settings)

    async with factory.create_embedding_model() as embedding_model, \
               factory.create_vector_store() as vector_store:

        # Skip if already populated (unless --force)
        if not force:
            stats = await vector_store.get_stats()
            if stats["document_count"] > 0:
                print(
                    f"  [{collection}] already has {stats['document_count']} chunks — "
                    f"skipping (use --force to re-ingest)"
                )
                return

        pipeline = factory.create_pipeline(
            embedding_model,
            vector_store,
            enrichers=strategy["enrichers"],
            extraction_mode=strategy["extraction_mode"],
        )

        t0 = time.time()
        results = await pipeline.ingest_directory(
            source_path,
            recursive=True,
            force=force,
        )

        elapsed = time.time() - t0
        total = sum(results.values())
        n_files = len(results)
        print(
            f"  [{collection}] {n_files} files → {total} chunks  "
            f"({elapsed:.0f}s)"
        )


async def run_ingestion(strategies: list[dict], source_path: Path, settings, force: bool) -> None:
    print("=" * 72)
    print("INGESTION PHASE")
    print(f"Source: {source_path}")
    print("=" * 72)

    for strategy in strategies:
        label = strategy["description"]
        print(f"\n  {strategy['collection']:12s}  {label}")
        await ingest_strategy(strategy, source_path, settings, force)

    print()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 3] + "..."


async def run_evaluation(strategies: list[dict], settings, k: int) -> None:
    print("=" * 72)
    print(f"EVALUATION PHASE  (k={k})")
    print("=" * 72)

    col_width = 55
    n_cols = len(strategies)
    table_width = col_width * n_cols + n_cols + 1

    async with _shared_embedding_model(settings) as embedding_model:
        for q in EVAL_QUERIES:
            query = q["query"]
            edition = q["edition"]
            edition_filter = {"edition": edition} if edition else None
            edition_tag = f"  [{edition}]" if edition else ""

            print()
            print("-" * table_width)
            print(f"Q: {query}{edition_tag}")
            print(f"   {q['label']} — {q['note']}")
            print("-" * table_width)

            # Header row
            header = " | ".join(s["label"].ljust(col_width) for s in strategies)
            print(f"| {header} |")
            print("-" * table_width)

            # Gather results from each collection
            all_results: list[list] = []
            for strategy in strategies:
                collection = strategy["collection"]
                rag_settings = settings.rag.model_copy(
                    update={"collection_name": collection}
                )
                factory = RAGComponents(rag_settings)
                try:
                    async with factory.create_vector_store() as vector_store:
                        engine = factory.create_engine(embedding_model, vector_store)
                        results = await engine.search(
                            query=query, k=k, filters=edition_filter
                        )
                        all_results.append(results)
                except Exception as exc:
                    print(f"  [{collection}] search failed: {exc}")
                    all_results.append([])

            # Print rank rows
            for rank in range(k):
                row_parts = []
                for results in all_results:
                    if rank < len(results):
                        r = results[rank]
                        preview = _truncate(r.text, col_width - 14)
                        cell = f"[{rank+1}] {r.score:.2f}  {preview}"
                    else:
                        cell = "(no result)"
                    row_parts.append(cell.ljust(col_width))
                print(f"| {' | '.join(row_parts)} |")

            # Print source row
            source_parts = []
            for results in all_results:
                if results:
                    src = results[0].citation[:col_width]
                else:
                    src = ""
                source_parts.append(src.ljust(col_width))
            print(f"  {'   '.join(s.strip() for s in source_parts)}")

        print()
        print("=" * 72)
        print("Evaluation complete.")
        print("=" * 72)


class _shared_embedding_model:
    """Async context manager that keeps one EmbeddingModel alive across all queries."""

    def __init__(self, settings):
        self._settings = settings
        self._factory = RAGComponents(settings.rag)
        self._ctx = None
        self._model = None

    async def __aenter__(self):
        self._ctx = self._factory.create_embedding_model()
        self._model = await self._ctx.__aenter__()
        return self._model

    async def __aexit__(self, *args):
        await self._ctx.__aexit__(*args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs with different strategies and compare retrieval quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "pdf",
        help="PDF source directory (default: data/raw/pdf)",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion — query existing collections only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if collections already have data",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Chunks to retrieve per query per collection (default: 3)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Alias for --skip-ingest",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    skip_ingest = args.skip_ingest or args.eval_only

    settings = load_settings()
    strategies = _make_strategies()

    if not skip_ingest:
        source = args.source
        if not source.exists():
            print(f"Error: source path does not exist: {source}", file=sys.stderr)
            sys.exit(1)
        await run_ingestion(strategies, source, settings, force=args.force)

    await run_evaluation(strategies, settings, k=args.k)


if __name__ == "__main__":
    asyncio.run(main())
