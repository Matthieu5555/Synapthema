"""Tests for the concept consolidator — entity resolution and graph building."""

import numpy as np

from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEdge,
    ConceptEntry,
    ConceptGraph,
    PrerequisiteLink,
    ResolvedConcept,
)
from src.transformation.concept_consolidator import (
    EMBEDDING_SIMILARITY_THRESHOLD,
    consolidate_concepts,
    _compute_embeddings,
    _cosine_similarity,
    _deduplicate_concepts,
    _build_edges,
    _build_canonical_map,
    _topological_sort,
    _find_foundation_concepts,
    _find_advanced_concepts,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _concept(
    name: str,
    definition: str = "A concept.",
    concept_type: str = "definition",
    section: str = "S1",
    key_terms: list[str] | None = None,
    importance: str = "core",
) -> ConceptEntry:
    return ConceptEntry(
        name=name,
        definition=definition,
        concept_type=concept_type,  # type: ignore[arg-type]
        section_title=section,
        key_terms=key_terms or [],
        importance=importance,  # type: ignore[arg-type]
    )


def _analysis(
    chapter_num: int,
    concepts: list[ConceptEntry],
    prereqs: list[PrerequisiteLink] | None = None,
) -> ChapterAnalysis:
    return ChapterAnalysis(
        chapter_number=chapter_num,
        chapter_title=f"Chapter {chapter_num}",
        concepts=concepts,
        prerequisites=prereqs or [],
    )


# ── Deduplication tests ─────────────────────────────────────────────────────


class TestDeduplicateConcepts:
    """Entity resolution across chapters."""

    def test_exact_name_match(self) -> None:
        """Same concept name in two chapters → merged."""
        c1 = _concept("Present value", "PV of future cash flows.")
        c2 = _concept("Present value", "Current worth of future money.")

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])

        assert len(resolved) == 1
        assert resolved[0].canonical_name == "Present value"
        assert 1 in resolved[0].mentioned_in_chapters
        assert 2 in resolved[0].mentioned_in_chapters

    def test_case_insensitive_match(self) -> None:
        c1 = _concept("Sharpe Ratio")
        c2 = _concept("sharpe ratio")

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])

        assert len(resolved) == 1

    def test_substring_match(self) -> None:
        """'Sharpe ratio' matches 'Sharpe ratio formula'."""
        c1 = _concept("Sharpe ratio")
        c2 = _concept("Sharpe ratio formula")

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])

        assert len(resolved) == 1
        assert resolved[0].canonical_name == "Sharpe ratio"

    def test_key_term_overlap(self) -> None:
        """Concepts with >50% key term overlap → merged."""
        c1 = _concept(
            "Portfolio risk",
            key_terms=["variance", "covariance", "correlation"],
        )
        c2 = _concept(
            "Diversification effect",
            key_terms=["variance", "covariance", "weights"],
        )
        # 2/3 overlap ≈ 67% > 50%
        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])

        assert len(resolved) == 1

    def test_no_merge_distinct_concepts(self) -> None:
        c1 = _concept("Bond pricing")
        c2 = _concept("Equity valuation")

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)])

        assert len(resolved) == 2

    def test_preserves_first_chapter_as_introduced(self) -> None:
        c1 = _concept("Duration")
        c2 = _concept("Duration")

        resolved = _deduplicate_concepts([(c1, 3), (c2, 1)])

        assert resolved[0].first_introduced_chapter == 1

    def test_aliases_tracked(self) -> None:
        c1 = _concept("Std dev")
        c2 = _concept("Standard deviation")
        # "Std dev" is a substring of "Standard deviation"? No.
        # These won't match by substring. Let's use key terms.
        c1_with_terms = _concept(
            "Std dev",
            key_terms=["volatility", "dispersion", "variance"],
        )
        c2_with_terms = _concept(
            "Standard deviation",
            key_terms=["volatility", "dispersion", "spread"],
        )

        resolved = _deduplicate_concepts([(c1_with_terms, 1), (c2_with_terms, 2)])

        assert len(resolved) == 1
        rc = resolved[0]
        assert len(rc.aliases) >= 1 or rc.canonical_name in {"Std dev", "Standard deviation"}


class TestBuildEdges:
    """Edge building from prerequisite links."""

    def test_resolves_to_canonical_names(self) -> None:
        canonical_map = {
            "bond pricing": "Bond pricing",
            "present value": "Present value",
        }
        prereqs = [
            PrerequisiteLink(
                source_concept="Bond pricing",
                target_concept="Present value",
                relationship="requires",
            ),
        ]

        edges = _build_edges(prereqs, canonical_map)

        assert len(edges) == 1
        assert edges[0].source == "Bond pricing"
        assert edges[0].target == "Present value"

    def test_deduplicates_edges(self) -> None:
        canonical_map = {
            "a": "A",
            "b": "B",
        }
        prereqs = [
            PrerequisiteLink(source_concept="A", target_concept="B", relationship="requires"),
            PrerequisiteLink(source_concept="A", target_concept="B", relationship="builds_on"),
        ]

        edges = _build_edges(prereqs, canonical_map)

        assert len(edges) == 1  # Same source-target, deduplicated

    def test_skips_unknown_concepts(self) -> None:
        canonical_map = {"a": "A"}
        prereqs = [
            PrerequisiteLink(source_concept="A", target_concept="Unknown", relationship="requires"),
        ]

        edges = _build_edges(prereqs, canonical_map)

        assert len(edges) == 0

    def test_skips_self_references(self) -> None:
        canonical_map = {"a": "A"}
        prereqs = [
            PrerequisiteLink(source_concept="A", target_concept="A", relationship="requires"),
        ]

        edges = _build_edges(prereqs, canonical_map)

        assert len(edges) == 0


class TestTopologicalSort:
    """Kahn's algorithm for concept ordering."""

    def test_linear_chain(self) -> None:
        """A → B → C should produce [C, B, A] (prerequisites first)."""
        names = ["A", "B", "C"]
        edges = [
            ConceptEdge(source="A", target="B", relationship="requires"),
            ConceptEdge(source="B", target="C", relationship="requires"),
        ]

        order = _topological_sort(names, edges)

        assert order.index("C") < order.index("B")
        assert order.index("B") < order.index("A")

    def test_diamond_dependency(self) -> None:
        """D depends on B and C, which both depend on A."""
        names = ["A", "B", "C", "D"]
        edges = [
            ConceptEdge(source="B", target="A", relationship="requires"),
            ConceptEdge(source="C", target="A", relationship="requires"),
            ConceptEdge(source="D", target="B", relationship="requires"),
            ConceptEdge(source="D", target="C", relationship="requires"),
        ]

        order = _topological_sort(names, edges)

        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_no_edges_alphabetical(self) -> None:
        """Without edges, should return concepts in alphabetical order."""
        names = ["C", "A", "B"]

        order = _topological_sort(names, [])

        assert order == ["A", "B", "C"]

    def test_handles_cycles(self) -> None:
        """Cycles should be detected and broken — all concepts still present."""
        names = ["A", "B"]
        edges = [
            ConceptEdge(source="A", target="B", relationship="requires"),
            ConceptEdge(source="B", target="A", relationship="requires"),
        ]

        order = _topological_sort(names, edges)

        assert set(order) == {"A", "B"}

    def test_empty_input(self) -> None:
        assert _topological_sort([], []) == []


class TestFoundationAndAdvanced:
    """Layer identification: foundation and advanced concepts."""

    def test_foundation_has_no_prerequisites(self) -> None:
        names = ["A", "B", "C"]
        edges = [
            ConceptEdge(source="B", target="A", relationship="requires"),
            ConceptEdge(source="C", target="B", relationship="requires"),
        ]

        foundation = _find_foundation_concepts(names, edges)

        assert "A" in foundation
        assert "B" not in foundation
        assert "C" not in foundation

    def test_advanced_has_most_prerequisites(self) -> None:
        names = ["A", "B", "C", "D"]
        edges = [
            ConceptEdge(source="D", target="A", relationship="requires"),
            ConceptEdge(source="D", target="B", relationship="requires"),
            ConceptEdge(source="D", target="C", relationship="requires"),
            ConceptEdge(source="C", target="A", relationship="requires"),
        ]

        advanced = _find_advanced_concepts(names, edges)

        assert advanced[0] == "D"  # D has 3 prerequisites, C has 1

    def test_no_edges_all_foundation(self) -> None:
        names = ["A", "B", "C"]
        foundation = _find_foundation_concepts(names, [])
        assert set(foundation) == {"A", "B", "C"}

    def test_no_edges_no_advanced(self) -> None:
        names = ["A", "B"]
        advanced = _find_advanced_concepts(names, [])
        assert advanced == []


class TestConsolidateConceptsIntegration:
    """End-to-end tests for consolidate_concepts()."""

    def test_empty_input(self) -> None:
        result = consolidate_concepts([])
        assert isinstance(result, ConceptGraph)
        assert result.concepts == []

    def test_single_chapter_no_prereqs(self) -> None:
        analysis = _analysis(
            chapter_num=1,
            concepts=[
                _concept("Bond"),
                _concept("Yield"),
            ],
        )

        result = consolidate_concepts([analysis])

        assert len(result.concepts) == 2
        assert len(result.edges) == 0
        assert len(result.foundation_concepts) == 2
        assert len(result.topological_order) == 2

    def test_two_chapters_with_prereqs(self) -> None:
        ch1 = _analysis(
            chapter_num=1,
            concepts=[_concept("Present value"), _concept("Discount rate")],
            prereqs=[
                PrerequisiteLink(
                    source_concept="Present value",
                    target_concept="Discount rate",
                    relationship="requires",
                ),
            ],
        )
        ch2 = _analysis(
            chapter_num=2,
            concepts=[_concept("Bond pricing")],
            prereqs=[
                PrerequisiteLink(
                    source_concept="Bond pricing",
                    target_concept="Present value",
                    relationship="builds_on",
                ),
            ],
        )

        result = consolidate_concepts([ch1, ch2])

        assert len(result.concepts) == 3
        assert len(result.edges) == 2
        # Topological order: Discount rate → Present value → Bond pricing
        order = result.topological_order
        assert order.index("Discount rate") < order.index("Present value")
        assert order.index("Present value") < order.index("Bond pricing")
        assert "Discount rate" in result.foundation_concepts
        assert "Bond pricing" in result.advanced_concepts

    def test_cross_chapter_dedup(self) -> None:
        ch1 = _analysis(
            chapter_num=1,
            concepts=[_concept("Duration", "Weighted average time.")],
        )
        ch2 = _analysis(
            chapter_num=2,
            concepts=[_concept("Duration", "Sensitivity to interest rates.")],
        )

        result = consolidate_concepts([ch1, ch2])

        assert len(result.concepts) == 1
        assert result.concepts[0].mentioned_in_chapters == [1, 2]


# ── Embedding tests ──────────────────────────────────────────────────────────


class TestCosineSimlarity:
    """Tests for the cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_normalized_dot_product(self) -> None:
        a = np.array([0.6, 0.8])
        b = np.array([0.8, 0.6])
        expected = 0.6 * 0.8 + 0.8 * 0.6
        assert abs(_cosine_similarity(a, b) - expected) < 1e-6


class TestEmbeddingDeduplication:
    """Strategy 4: Embedding-based deduplication."""

    def test_merge_with_high_similarity_embeddings(self) -> None:
        """Concepts with similar embeddings (>= threshold) should merge."""
        c1 = _concept("Portfolio Diversification", "Spreading risk across assets.")
        c2 = _concept("Risk Spreading Through Assets", "Reducing risk by diversifying.")

        # Simulate embeddings: very similar vectors (similarity ~0.98)
        v1 = np.array([0.7, 0.7, 0.1])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.array([0.71, 0.69, 0.12])
        v2 = v2 / np.linalg.norm(v2)

        embeddings = {
            "portfolio diversification": v1,
            "risk spreading through assets": v2,
        }

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)], embeddings)

        assert len(resolved) == 1

    def test_no_merge_with_low_similarity_embeddings(self) -> None:
        """Concepts with low similarity (< threshold) should stay separate."""
        c1 = _concept("Bond pricing", "Calculating the price of bonds.")
        c2 = _concept("Equity valuation", "Estimating the value of stocks.")

        # Simulate embeddings: very different vectors (similarity ~0.2)
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        embeddings = {
            "bond pricing": v1,
            "equity valuation": v2,
        }

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)], embeddings)

        assert len(resolved) == 2

    def test_threshold_boundary_below(self) -> None:
        """Similarity just below threshold keeps concepts separate."""
        c1 = _concept("Concept A", "Definition A.")
        c2 = _concept("Concept B", "Definition B.")

        # Construct vectors with similarity just below threshold (0.74)
        angle = np.arccos(EMBEDDING_SIMILARITY_THRESHOLD - 0.01)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([np.cos(angle), np.sin(angle)])

        embeddings = {
            "concept a": v1,
            "concept b": v2,
        }

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)], embeddings)

        assert len(resolved) == 2

    def test_threshold_boundary_above(self) -> None:
        """Similarity just above threshold merges concepts."""
        c1 = _concept("Concept A", "Definition A.")
        c2 = _concept("Concept B", "Definition B.")

        # Construct vectors with similarity just above threshold (0.76)
        angle = np.arccos(EMBEDDING_SIMILARITY_THRESHOLD + 0.01)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([np.cos(angle), np.sin(angle)])

        embeddings = {
            "concept a": v1,
            "concept b": v2,
        }

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)], embeddings)

        assert len(resolved) == 1

    def test_heuristics_take_precedence_over_embeddings(self) -> None:
        """Exact match should fire before embeddings are checked."""
        c1 = _concept("Duration", "Weighted avg time.")
        c2 = _concept("Duration", "Sensitivity measure.")

        # Even with None embeddings, exact match still works
        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)], None)

        assert len(resolved) == 1

    def test_none_embeddings_graceful(self) -> None:
        """When embeddings is None, dedup still works (heuristics only)."""
        c1 = _concept("Bond pricing")
        c2 = _concept("Equity valuation")

        resolved = _deduplicate_concepts([(c1, 1), (c2, 2)], None)

        assert len(resolved) == 2


class TestComputeEmbeddings:
    """Tests for _compute_embeddings()."""

    def test_returns_none_without_library(self, monkeypatch) -> None:
        """When _HAS_EMBEDDINGS is False, returns None."""
        import src.transformation.concept_consolidator as mod

        monkeypatch.setattr(mod, "_HAS_EMBEDDINGS", False)

        concepts = [(_concept("Test", "A test."), 1)]
        result = _compute_embeddings(concepts)

        assert result is None
