from __future__ import annotations

import re
import uuid

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Stable namespace for mapping string chunk ids to Qdrant point UUIDs.
_QDRANT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def sql_table_identifier(name: str) -> str:
    """Return a safe SQL table reference from vector_store.collection."""
    if _IDENT_RE.fullmatch(name):
        return name
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def sql_index_slug(name: str) -> str:
    """Alphanumeric slug for derived index names (unquoted identifiers)."""
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not slug or slug[0].isdigit():
        slug = f"t_{slug}"
    return slug


def qdrant_point_id(chunk_id: str) -> str:
    """Map an arbitrary string chunk id to a Qdrant-compatible UUID."""
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, chunk_id))


def normalize_filters(filters: dict | None) -> dict | None:
    """Treat empty filter dicts as no filter (Chroma rejects `{}`)."""
    if not filters:
        return None
    return filters