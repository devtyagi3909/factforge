"""
Production FastAPI Gateway for FactForge Financial Document Reconciliation Engine.
Exposes:
- /ingest: Streaming ingestion of PDF and text financial filings
- /reconcile: O(M log N) cross-document contradiction detection and reconciliation
- /graph: Knowledge graph serialization (facts, corroborations, contradictions)
- /health: Service health and readiness probe
"""

import os
import tempfile
from typing import Dict, Any, List, Optional
from functools import wraps
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel, Field
from core.parser import fact_layer, classify_provider_error

_HAS_LANGFUSE = False
try:
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse.decorators import observe as _observe
        _HAS_LANGFUSE = True
except ImportError:
    _HAS_LANGFUSE = False


def observe(*args, **kwargs):
    if _HAS_LANGFUSE:
        return _observe(*args, **kwargs)

    def decorator(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            return fn(*a, **kw)
        return wrapper

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    return decorator


app = FastAPI(
    title="FactForge API - Financial Document Grounding & Reconciliation",
    description="O(M log N) incremental fact extraction, verbatim excerpt grounding, and contradiction detection",
    version="1.2.0"
)


class TextIngestRequest(BaseModel):
    document_name: str
    document_text: str


class ReconcileRequest(BaseModel):
    entity_name: Optional[str] = None
    force_refresh: bool = False


@app.post("/ingest")
@observe(name="api_ingest_document")
async def ingest_document(
    file: Optional[UploadFile] = File(default=None),
    document_name: Optional[str] = Form(default=None),
    document_text: Optional[str] = Form(default=None),
):
    """
    Ingest a financial filing (PDF or TXT) into the incremental fact ledger.
    Enforces strict 100% verbatim excerpt substring validation.
    """
    if file:
        filename = file.filename or "upload.pdf"
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    elif document_text and document_name:
        filename = document_name
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(document_text)
            tmp_path = tmp.name
    else:
        raise HTTPException(status_code=400, detail="Must provide either a file upload or document_name + document_text.")

    try:
        result = fact_layer.process_document(tmp_path, filename)
        facts = fact_layer.get_facts()
        return {
            "status": result.get("status", "completed"),
            "document": filename,
            "total_facts_in_ledger": len(facts),
            "result": result
        }
    except Exception as exc:
        err = classify_provider_error(exc)
        raise HTTPException(status_code=500, detail=err.get("message", str(exc)))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/reconcile")
@observe(name="api_reconcile_contradictions")
def reconcile_contradictions(req: ReconcileRequest):
    """
    Run O(M log N) incremental contradiction detection across ingested documents.
    """
    reasoning = fact_layer.run_reasoning()
    contradictions = reasoning.get("contradictions", [])
    corroborations = reasoning.get("corroborations", [])

    if req.entity_name:
        target = req.entity_name.lower()
        contradictions = [c for c in contradictions if target in str(c).lower()]
        corroborations = [c for c in corroborations if target in str(c).lower()]

    return {
        "entity_filter": req.entity_name,
        "contradictions_count": len(contradictions),
        "corroborations_count": len(corroborations),
        "contradictions": contradictions,
        "corroborations": corroborations,
        "status": "SUCCESS"
    }


@app.get("/graph")
@observe(name="api_get_knowledge_graph")
def get_knowledge_graph():
    """
    Retrieve nodes, edges, and citation metadata for UI or downstream API rendering.
    """
    graph_data = fact_layer.build_graph()
    return {
        "status": "SUCCESS",
        "nodes": graph_data.get("nodes", []),
        "edges": graph_data.get("edges", [])
    }


@app.post("/demo")
def load_demo_dataset():
    """
    Load verified synthetic evaluation dataset for zero-key local testing.
    """
    demo_result = fact_layer.load_demo()
    return {
        "status": "SUCCESS",
        "message": "Loaded 5 canonical demo facts and resolved contradictions",
        "demo": demo_result
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "factforge-reconciliation-engine",
        "version": "1.2.0",
        "engine": "Inverted Lexical Index + Deterministic Scale Canonicalizer"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
