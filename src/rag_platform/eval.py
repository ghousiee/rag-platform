from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from rag_platform.config import Settings
from rag_platform.generation import query
from rag_platform.models import QueryResult

PASS_KEYWORD_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class GoldenItem(BaseModel):
    question: str
    expected_keywords: list[str] | None = None
    reference_answer: str | None = None


class EvalCaseResult(BaseModel):
    question: str
    answer: str
    keyword_score: float
    overlap_score: float | None
    passed: bool


class EvalReport(BaseModel):
    results: list[EvalCaseResult]
    mean_keyword_score: float


def token_jaccard(a: str, b: str) -> float:
    tokens_a = set(_TOKEN_RE.findall(a.lower()))
    tokens_b = set(_TOKEN_RE.findall(b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _keyword_score(answer: str, keywords: list[str] | None) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords)


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(GoldenItem.model_validate(json.loads(line)))
    return items


def run_eval(
    settings: Settings,
    *,
    query_fn: Callable[..., QueryResult] = query,
) -> EvalReport:
    golden_path = settings.eval.golden_path
    items = load_golden(golden_path)
    results: list[EvalCaseResult] = []

    for item in items:
        result = query_fn(item.question, settings)
        kw_score = _keyword_score(result.answer, item.expected_keywords)
        overlap: float | None = None
        if item.reference_answer is not None:
            overlap = token_jaccard(result.answer, item.reference_answer)
        passed = kw_score >= PASS_KEYWORD_THRESHOLD
        results.append(
            EvalCaseResult(
                question=item.question,
                answer=result.answer,
                keyword_score=kw_score,
                overlap_score=overlap,
                passed=passed,
            )
        )

    mean_kw = sum(r.keyword_score for r in results) / len(results) if results else 0.0
    return EvalReport(results=results, mean_keyword_score=mean_kw)


def write_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )