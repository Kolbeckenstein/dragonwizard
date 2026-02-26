"""
WeightedHeadingEnricher — heading injection with confidence scores.

Identical to StatisticalHeadingEnricher except the heading prefix includes
the confidence score: "[Section: Orc (0.87)]\\n" instead of "[Section: Orc]\\n".

The confidence score (0.0 to 1.0) is derived from the font-size ratio:
    confidence = min(1.0, font_ratio / 2.0)

Useful for downstream analysis — you can filter or weight chunks by how
confident the enricher was about each heading assignment.
"""

from dragonwizard.rag.sources.pdf.statistical_enricher import StatisticalHeadingEnricher, _HeadingCandidate


class WeightedHeadingEnricher(StatisticalHeadingEnricher):
    """
    Heading enricher that includes confidence scores in the injected prefix.

    Prefix format: "[Section: <heading> (<confidence:.2f>)]\\n"
    Example:        "[Section: Orc (0.70)]\\n"

    Everything else (detection thresholds, max_section_gap, etc.) is identical
    to StatisticalHeadingEnricher.
    """

    def _format_prefix(self, heading: _HeadingCandidate) -> str:
        """Format heading prefix including the confidence score."""
        return f"[Section: {heading.text} ({heading.confidence:.2f})]\n"
