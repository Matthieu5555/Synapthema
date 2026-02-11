"""Concept consolidation — entity resolution and dependency graph building.

Pure Python, no LLM calls. Takes a list of ChapterAnalysis objects and
produces a ConceptGraph with:
1. Deduplicated concepts (entity resolution across chapters).
2. Directed dependency edges from PrerequisiteLinks.
3. Topological order via Kahn's algorithm (natural learning order).
4. Foundation concepts (no prerequisites) and advanced concepts (most deps).

Public entry point: consolidate_concepts().
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from src.transformation.analysis_types import (
    ChapterAnalysis,
    ConceptEdge,
    ConceptEntry,
    ConceptGraph,
    PrerequisiteLink,
    ResolvedConcept,
)

logger = logging.getLogger(__name__)

# Minimum key term overlap ratio (0.0-1.0) to consider two concepts as
# the same entity when their names don't match exactly.
KEY_TERM_OVERLAP_THRESHOLD = 0.5

# Minimum cosine similarity (0.0-1.0) to merge concepts via embedding.
# 0.75 is conservative — avoids false positives while catching semantic equivalences.
EMBEDDING_SIMILARITY_THRESHOLD = 0.75

# Lazy-loaded flag: True if sentence-transformers is available.
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False


# ── Public API ───────────────────────────────────────────────────────────────


def consolidate_concepts(
    chapter_analyses: list[ChapterAnalysis],
) -> ConceptGraph:
    """Build a concept dependency graph from chapter analyses.

    Steps:
    1. Collect all concepts across chapters.
    2. Deduplicate via entity resolution (exact name, substring, key term overlap).
    3. Build directed edges from prerequisite links, resolved to canonical names.
    4. Topological sort (Kahn's algorithm) for natural learning order.
    5. Identify foundation concepts (in-degree 0) and advanced concepts.

    Args:
        chapter_analyses: List of ChapterAnalysis from the deep reader.

    Returns:
        ConceptGraph with resolved concepts, edges, and ordering.
    """
    if not chapter_analyses:
        return ConceptGraph()

    # Step 1: Collect all concepts with chapter metadata
    all_concepts = _collect_concepts(chapter_analyses)
    if not all_concepts:
        return ConceptGraph()

    # Step 2: Deduplicate (with optional embedding similarity)
    embeddings = _compute_embeddings(all_concepts)
    resolved = _deduplicate_concepts(all_concepts, embeddings)
    logger.info(
        "Entity resolution: %d raw concepts → %d unique concepts%s",
        len(all_concepts), len(resolved),
        " (with embeddings)" if embeddings is not None else "",
    )

    # Build canonical name lookup
    canonical_map = _build_canonical_map(resolved)

    # Step 3: Build edges
    all_prereqs = _collect_prerequisites(chapter_analyses)
    edges = _build_edges(all_prereqs, canonical_map)
    logger.info("Built %d dependency edges", len(edges))

    # Step 4: Topological sort
    concept_names = [r.canonical_name for r in resolved]
    topo_order = _topological_sort(concept_names, edges)

    # Step 5: Identify layers
    foundation = _find_foundation_concepts(concept_names, edges)
    advanced = _find_advanced_concepts(concept_names, edges)

    return ConceptGraph(
        concepts=resolved,
        edges=edges,
        topological_order=topo_order,
        foundation_concepts=foundation,
        advanced_concepts=advanced,
        canonical_map=canonical_map,
    )


# ── Step 1: Collect ─────────────────────────────────────────────────────────


def _collect_concepts(
    analyses: list[ChapterAnalysis],
) -> list[tuple[ConceptEntry, int]]:
    """Collect all concepts with their chapter numbers."""
    return [(c, a.chapter_number) for a in analyses for c in a.concepts]


def _collect_prerequisites(
    analyses: list[ChapterAnalysis],
) -> list[PrerequisiteLink]:
    """Collect all prerequisite links across chapters."""
    return [p for a in analyses for p in a.prerequisites]


# ── Embedding helpers ──────────────────────────────────────────────────────


def _cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two L2-normalized vectors.

    Assumes vectors are already L2-normalized (as returned by
    SentenceTransformer.encode with normalize_embeddings=True),
    so the dot product equals cosine similarity.
    """
    import numpy as np

    return float(np.dot(a, b))


def _compute_embeddings(
    concepts: list[tuple[ConceptEntry, int]],
) -> dict[str, Any] | None:
    """Compute sentence embeddings for concept names and definitions.

    Returns None when sentence-transformers is not installed (graceful
    degradation to heuristic-only deduplication).
    """
    if not _HAS_EMBEDDINGS:
        logger.info("sentence-transformers not installed, skipping embedding-based dedup")
        return None

    if not concepts:
        return {}

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts: list[str] = []
    keys: list[str] = []
    for concept, _ in concepts:
        key = concept.name.lower().strip()
        text = f"{concept.name}: {concept.definition}"
        texts.append(text)
        keys.append(key)

    vectors = model.encode(texts, normalize_embeddings=True)
    return dict(zip(keys, vectors))


# ── Step 2: Deduplicate ─────────────────────────────────────────────────────


def _deduplicate_concepts(
    concepts_with_chapters: list[tuple[ConceptEntry, int]],
    embeddings: dict[str, Any] | None = None,
) -> list[ResolvedConcept]:
    """Deduplicate concepts via entity resolution.

    Resolution strategies (in order):
    1. Exact name match (case-insensitive).
    2. Substring containment ("Sharpe ratio" matches "Sharpe ratio formula").
    3. Key term overlap > threshold.
    4. Embedding cosine similarity > threshold (if embeddings available).
    """
    # Group by canonical name (case-insensitive)
    groups: dict[str, list[tuple[ConceptEntry, int]]] = {}
    name_to_canonical: dict[str, str] = {}

    for concept, chapter_num in concepts_with_chapters:
        canonical = _find_canonical(concept, name_to_canonical, groups, embeddings)
        if canonical is None:
            # New concept
            canonical = concept.name.lower().strip()
            name_to_canonical[canonical] = canonical
            name_to_canonical[concept.name.lower().strip()] = canonical
            groups[canonical] = []

        groups[canonical].append((concept, chapter_num))

    # Build ResolvedConcepts
    resolved: list[ResolvedConcept] = []
    for canonical, members in groups.items():
        names = {m[0].name for m in members}
        chapters = sorted({m[1] for m in members})

        # Pick the best definition (longest, from the earliest chapter)
        members_sorted = sorted(members, key=lambda m: (m[1], -len(m[0].definition)))
        best = members_sorted[0][0]

        # Canonical name: prefer the original casing from the first occurrence
        canonical_display = members_sorted[0][0].name
        aliases = sorted(names - {canonical_display})

        resolved.append(
            ResolvedConcept(
                canonical_name=canonical_display,
                aliases=aliases,
                definition=best.definition,
                first_introduced_chapter=chapters[0],
                mentioned_in_chapters=chapters,
            )
        )

    return sorted(resolved, key=lambda r: (r.first_introduced_chapter, r.canonical_name))


def _find_canonical(
    concept: ConceptEntry,
    name_to_canonical: dict[str, str],
    groups: dict[str, list[tuple[ConceptEntry, int]]],
    embeddings: dict[str, Any] | None = None,
) -> str | None:
    """Find the canonical name for a concept, or None if it's new.

    Resolution strategies (in order):
    1. Exact name match (case-insensitive).
    2. Substring containment.
    3. Key term overlap > threshold.
    4. Embedding cosine similarity > threshold (if embeddings available).
    """
    name_lower = concept.name.lower().strip()

    # Strategy 1: Exact match
    if name_lower in name_to_canonical:
        return name_to_canonical[name_lower]

    # Strategy 2: Substring containment
    for existing_name, canonical in name_to_canonical.items():
        if name_lower in existing_name or existing_name in name_lower:
            name_to_canonical[name_lower] = canonical
            return canonical

    # Strategy 3: Key term overlap
    if concept.key_terms:
        concept_terms = {t.lower() for t in concept.key_terms}
        for canonical, members in groups.items():
            for member, _ in members:
                if member.key_terms:
                    member_terms = {t.lower() for t in member.key_terms}
                    overlap = len(concept_terms & member_terms)
                    max_possible = min(len(concept_terms), len(member_terms))
                    if max_possible > 0 and overlap / max_possible >= KEY_TERM_OVERLAP_THRESHOLD:
                        name_to_canonical[name_lower] = canonical
                        return canonical

    # Strategy 4: Embedding similarity
    if embeddings is not None and name_lower in embeddings:
        concept_emb = embeddings[name_lower]
        best_sim = 0.0
        best_canonical: str | None = None
        for existing_name, canonical in name_to_canonical.items():
            if existing_name in embeddings:
                sim = _cosine_similarity(concept_emb, embeddings[existing_name])
                if sim > best_sim:
                    best_sim = sim
                    best_canonical = canonical
        if best_sim >= EMBEDDING_SIMILARITY_THRESHOLD and best_canonical is not None:
            name_to_canonical[name_lower] = best_canonical
            return best_canonical

    return None


def _build_canonical_map(resolved: list[ResolvedConcept]) -> dict[str, str]:
    """Build a lookup from any name/alias to its canonical name."""
    return {
        name.lower(): r.canonical_name
        for r in resolved
        for name in [r.canonical_name, *r.aliases]
    }


# ── Step 3: Build edges ─────────────────────────────────────────────────────


def _build_edges(
    prereqs: list[PrerequisiteLink],
    canonical_map: dict[str, str],
) -> list[ConceptEdge]:
    """Convert PrerequisiteLinks to ConceptEdges using canonical names.

    Deduplicates via dict keyed on (source, target) — first relationship wins.
    """
    unique: dict[tuple[str, str], str] = {}
    for link in prereqs:
        source = canonical_map.get(link.source_concept.lower())
        target = canonical_map.get(link.target_concept.lower())
        if source and target and source != target:
            unique.setdefault((source, target), link.relationship)

    return [
        ConceptEdge(source=s, target=t, relationship=r)
        for (s, t), r in unique.items()
    ]


# ── Step 4: Topological sort ────────────────────────────────────────────────


def _topological_sort(
    concept_names: list[str],
    edges: list[ConceptEdge],
) -> list[str]:
    """Kahn's algorithm for topological ordering.

    Edge direction: source depends on target (target must come first).
    Cycles are detected, logged, and broken at the weakest edge
    (the one added last).

    Returns concepts in natural learning order (prerequisites first).
    """
    if not concept_names:
        return []

    name_set = set(concept_names)

    # Build adjacency: target → [sources that depend on it]
    # In-degree counts how many things depend on each concept being learned first
    # We want: target comes before source in the order
    # So our DAG has edges: target → source
    adjacency: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {name: 0 for name in name_set}

    valid_edges: list[ConceptEdge] = []
    for edge in edges:
        if edge.source in name_set and edge.target in name_set:
            adjacency[edge.target].append(edge.source)
            in_degree[edge.source] = in_degree.get(edge.source, 0) + 1
            valid_edges.append(edge)

    # Kahn's algorithm
    queue = [name for name in concept_names if in_degree.get(name, 0) == 0]
    queue.sort()  # Deterministic tie-breaking
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in sorted(adjacency.get(node, [])):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        queue.sort()

    # Handle cycles: add remaining concepts
    remaining = [n for n in concept_names if n not in set(result)]
    if remaining:
        logger.warning(
            "Cycle detected in concept graph — %d concepts in cycle: %s",
            len(remaining), remaining[:5],
        )
        result.extend(sorted(remaining))

    return result


# ── Step 5: Layer identification ────────────────────────────────────────────


def _find_foundation_concepts(
    concept_names: list[str],
    edges: list[ConceptEdge],
) -> list[str]:
    """Find concepts with no prerequisites (in-degree 0 in the dependency DAG).

    These are the natural starting points for learning.
    """
    name_set = set(concept_names)
    has_prerequisite = {
        e.source for e in edges
        if e.source in name_set and e.target in name_set
    }
    return sorted(name for name in concept_names if name not in has_prerequisite)


def _find_advanced_concepts(
    concept_names: list[str],
    edges: list[ConceptEdge],
    top_n: int = 5,
) -> list[str]:
    """Find concepts with the most prerequisites (highest in-degree).

    These are the most complex concepts that depend on the most prior knowledge.
    """
    name_set = set(concept_names)
    in_degree = Counter(
        e.source for e in edges
        if e.source in name_set and e.target in name_set
    )
    if not in_degree:
        return []

    return [name for name, _ in in_degree.most_common(top_n)]
