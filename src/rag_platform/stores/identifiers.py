from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


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