from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from rag_platform.api.app import create_app
from rag_platform.config import load_settings
from rag_platform.eval import run_eval, write_report
from rag_platform.generation import query as run_query
from rag_platform.ingest import ingest_path
from rag_platform.models import EmptyIndexError

app = typer.Typer(help="RAG platform CLI")


@app.command()
def ingest(
    path: Annotated[Path | None, typer.Option("--path", help="Directory to ingest")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="YAML config path")] = None,
) -> None:
    settings = load_settings(config)
    target = path if path is not None else settings.paths.raw
    try:
        result = ingest_path(target, settings)
    except (ConnectionError, OSError) as exc:
        typer.echo(f"Connection error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Ingest complete: files={result.files}, chunks={result.chunks}")
    for err in result.errors:
        typer.echo(err, err=True)
    if result.errors:
        raise typer.Exit(code=1)


@app.command()
def query(
    question: str,
    config: Annotated[Path | None, typer.Option("--config", help="YAML config path")] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", help="Override retrieval top_k")] = None,
) -> None:
    settings = load_settings(config)
    try:
        result = run_query(question, settings, top_k=top_k)
    except EmptyIndexError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except (ConnectionError, OSError) as exc:
        typer.echo(f"Connection error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(result.answer)
    for citation in result.citations:
        preview = citation.text.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        typer.echo(
            f"  [{citation.score:.3f}] {citation.source}: {preview}",
        )


@app.command(name="eval")
def eval_cmd(
    config: Annotated[Path | None, typer.Option("--config", help="YAML config path")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="JSON report output path")] = None,
) -> None:
    settings = load_settings(config)
    report = run_eval(settings)
    out_path = out if out is not None else Path("evals/report.json")
    write_report(report, out_path)
    typer.echo(
        f"Eval complete: cases={len(report.results)}, "
        f"mean_keyword_score={report.mean_keyword_score:.3f}, written={out_path}"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    config: Annotated[Path | None, typer.Option("--config", help="YAML config path")] = None,
) -> None:
    settings = load_settings(config)
    fastapi_app = create_app(settings)
    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()