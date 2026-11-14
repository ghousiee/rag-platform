from __future__ import annotations

from typing import Any

from rag_platform.config import VectorStoreConfig
from rag_platform.stores.base import SearchHit

_TABLE = "rag_vectors"


class PgVectorStore:
    def __init__(self, config: VectorStoreConfig, dsn: str) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._config = config
        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        import psycopg

        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{_TABLE}_source
                ON {_TABLE} ((metadata->>'source'))
                """
            )

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        import json

        with self._conn.cursor() as cur:
            for row_id, text, embedding, meta in zip(
                ids, texts, embeddings, metadatas
            ):
                cur.execute(
                    f"""
                    INSERT INTO {_TABLE} (id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (row_id, text, embedding, json.dumps(meta)),
                )

    def similarity_search(
        self,
        vector: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[SearchHit]:
        where, filter_params = _build_where(filters)
        params = [vector, *filter_params, vector, k]
        sql = f"""
            SELECT id, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {_TABLE}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            SearchHit(
                id=row[0],
                text=row[1],
                score=float(row[3]),
                metadata=dict(row[2] or {}),
            )
            for row in rows
        ]

    def delete_by_source(self, source: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {_TABLE} WHERE metadata->>'source' = %s",
                (source,),
            )

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {_TABLE}")
            row = cur.fetchone()
        return int(row[0]) if row else 0


def _build_where(filters: dict | None) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for key, value in filters.items():
        clauses.append(f"metadata->>%s = %s")
        params.extend([key, str(value)])
    return "WHERE " + " AND ".join(clauses), params