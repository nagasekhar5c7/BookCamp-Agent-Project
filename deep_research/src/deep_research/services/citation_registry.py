"""Citation registry — the single source of truth for source ids.

The registry is a deduplicated mapping from **canonical URL** to a
**stable integer id**. Every claim the system produces references one
or more of these ids so that the final Word document can render
consistent inline markers (e.g. ``[3]``) and a References section keyed
by the same integers.

Design notes:

- **Deduplication key is the URL.** Two Tavily hits with different
  titles but the same URL collapse to one citation. This means the
  first title/snippet wins; later hits are silently dropped.
- **Ids are monotonically increasing from 1.** Starting at 1 matches
  how humans read "[1]" in the final report and is the convention the
  LLM is prompted with.
- **Instances are per-job.** The graph constructs a fresh registry at
  the start of the ``research`` stage; its final state is serialised
  into :attr:`~deep_research.graph.state.ResearchState.citations` and
  passed to synthesis.
- **Not thread-safe.** Sub-tasks are executed sequentially in v1 so
  there is no contention. If the research loop ever goes concurrent,
  add a lock here.
"""

from __future__ import annotations

from collections.abc import Iterator

from deep_research.models.citation import Citation


class CitationRegistry:
    """Deduplicating int-id store for :class:`Citation` objects."""

    def __init__(self) -> None:
        # url -> id mapping drives dedup. id -> citation is what callers consume.
        self._url_to_id: dict[str, int] = {}
        self._id_to_citation: dict[int, Citation] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------ API

    def register(self, citation: Citation) -> int:
        """Insert ``citation`` if new and return its stable id.

        If the URL is already registered, returns the existing id
        without overwriting the stored citation — first writer wins.
        """
        key = str(citation.url)
        existing = self._url_to_id.get(key)
        if existing is not None:
            return existing

        new_id = self._next_id
        self._next_id += 1
        self._url_to_id[key] = new_id
        self._id_to_citation[new_id] = citation
        return new_id

    def get(self, citation_id: int) -> Citation | None:
        """Return the citation for ``citation_id`` or ``None`` if unknown."""
        return self._id_to_citation.get(citation_id)

    def contains(self, citation_id: int) -> bool:
        """True iff ``citation_id`` is a valid registered id."""
        return citation_id in self._id_to_citation

    def items(self) -> Iterator[tuple[int, Citation]]:
        """Iterate over ``(id, citation)`` pairs in registration order."""
        return iter(sorted(self._id_to_citation.items()))

    def as_dict(self) -> dict[int, Citation]:
        """Return a shallow copy of the id → citation map.

        Used when serialising the registry into the graph state at the
        end of the ``research`` stage.
        """
        return dict(self._id_to_citation)

    def __len__(self) -> int:
        return len(self._id_to_citation)

    # ------------------------------------------------------------- Factories

    @classmethod
    def from_dict(cls, mapping: dict[int, Citation]) -> "CitationRegistry":
        """Rebuild a registry from a previously serialised id → citation map.

        Used by the synthesize node, which needs a live registry to
        format the citation map for the Lead prompt but receives only
        the serialised dict on state.
        """
        reg = cls()
        if not mapping:
            return reg
        for cid, citation in sorted(mapping.items()):
            reg._id_to_citation[cid] = citation
            reg._url_to_id[str(citation.url)] = cid
        reg._next_id = max(mapping) + 1
        return reg
