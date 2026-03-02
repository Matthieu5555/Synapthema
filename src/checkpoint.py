"""Generic checkpoint load/save for pipeline stages.

Provides two pure functions for checkpoint persistence. Any checkpoint is
either valid typed data or None — invalid files are silently treated as
missing so the pipeline re-runs the stage.

Used by: pipeline.py (all stages).
Dependencies: pydantic (TypeAdapter for generic deserialization).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")

logger = logging.getLogger(__name__)


def load_checkpoint(path: Path, data_type: type[T]) -> T | None:
    """Load a typed checkpoint from a JSON file.

    Returns the deserialized data on success, or None if the file is
    missing, unreadable, or fails validation. Never raises.
    """
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        adapter = TypeAdapter(data_type)
        return adapter.validate_json(raw)
    except Exception:
        logger.debug("Invalid checkpoint %s, will re-run stage", path)
        return None


def save_checkpoint(path: Path, data: Any) -> None:
    """Save data as a JSON checkpoint, creating parent directories.

    Handles both Pydantic BaseModel instances and arbitrary types
    serializable via TypeAdapter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        json_str = data.model_dump_json(indent=2)
    else:
        adapter = TypeAdapter(type(data))
        json_bytes = adapter.dump_json(data, indent=2)
        json_str = json_bytes.decode("utf-8")
    path.write_text(json_str, encoding="utf-8")


def save_checkpoint_raw(path: Path, data: Any) -> None:
    """Save data as JSON using json.dumps for types needing custom serialization.

    Use this for frozen dataclasses or composite structures that need
    dataclasses.asdict() or custom dict building before serialization.
    The caller provides a JSON-serializable dict/list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
