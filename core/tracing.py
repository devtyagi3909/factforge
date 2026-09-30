"""
Langfuse Observability & Tracing Integration for Fact Knowledge Layer.
Provides production tracing for LLM calls, fact extraction spans, and evaluation scores.
Falls back to clean local structured logging when LANGFUSE API keys are omitted.
"""

import os
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("fact_layer.tracing")

_langfuse_client = None
_has_langfuse = False

try:
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        _has_langfuse = True
        logger.info("Langfuse tracing initialized successfully.")
except Exception as e:
    logger.warning(f"Langfuse import/init skipped: {e}. Running in local trace mode.")


def trace_fact_extraction(doc_name: str, chunk_index: int, total_chunks: int, model: str):
    """Context manager or span wrapper for chunk fact extraction."""
    start_time = time.time()
    trace_id = f"{doc_name}-chunk-{chunk_index}"
    
    trace_obj = None
    if _has_langfuse and _langfuse_client:
        try:
            trace_obj = _langfuse_client.trace(
                name="extract_facts_from_chunk",
                metadata={"doc": doc_name, "chunk": chunk_index, "total": total_chunks, "model": model}
            )
        except Exception:
            pass

    return TraceContext(trace_obj, trace_id, start_time)


class TraceContext:
    def __init__(self, trace_obj, trace_id: str, start_time: float):
        self.trace_obj = trace_obj
        self.trace_id = trace_id
        self.start_time = start_time

    def log_result(self, facts_extracted: int, tokens_used: int, error: Optional[str] = None):
        duration = time.time() - self.start_time
        if self.trace_obj:
            try:
                self.trace_obj.span(
                    name="chunk_completion",
                    output={"facts_count": facts_extracted, "tokens": tokens_used, "error": error},
                    metadata={"duration_sec": round(duration, 3)}
                )
            except Exception:
                pass
        logger.info(f"Trace [{self.trace_id}] extracted {facts_extracted} facts in {duration:.2f}s ({tokens_used} tokens).")
