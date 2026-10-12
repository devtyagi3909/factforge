"""
FastMCP Enterprise Server for FactForge Financial Document Reconciliation.
Exposes Model Context Protocol (MCP) tools:
- extract_filing_facts: Extracts grounded facts from unstructured SEC filings / financial reports
- detect_corpus_contradictions: Detects conflicting metrics, restatements, and temporal discrepancies
- query_knowledge_graph: Queries cross-document entity relationships and financial metric ledgers
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.parser import fact_layer

try:
    from fastmcp import FastMCP
    mcp = FastMCP("FactForgeServer")
except ImportError:
    class FastMCPShim:
        def __init__(self, name: str = "FactForgeServer"):
            self.name = name
            self.tools = {}

        def tool(self, **kwargs):
            def decorator(func):
                self.tools[func.__name__] = func
                return func
            return decorator

    mcp = FastMCPShim("FactForgeServer")


@mcp.tool()
def extract_filing_facts(document_text: str, document_name: str = "filing.txt") -> Dict[str, Any]:
    """
    Extract grounded financial and operational facts from an SEC filing or earnings release text.
    Enforces 100% verbatim excerpt substring invariant against the input document.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(document_text)
        temp_path = f.name

    try:
        res = fact_layer.process_document(temp_path, document_name)
        facts = fact_layer.get_facts()
        return {
            "document_name": document_name,
            "status": res.get("status", "completed"),
            "facts_extracted": len(facts),
            "facts": [f.model_dump() for f in facts]
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@mcp.tool()
def detect_corpus_contradictions(entity_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan the cross-document knowledge ledger for numerical or temporal contradictions
    concerning a target entity or financial metric (e.g. ARR, Revenue, CapEx).
    """
    reasoning = fact_layer.run_reasoning()
    contradictions = reasoning.get("contradictions", [])
    if entity_name:
        contradictions = [
            c for c in contradictions
            if entity_name.lower() in str(c).lower()
        ]

    return {
        "entity_filter": entity_name,
        "total_contradictions": len(contradictions),
        "contradictions": contradictions,
        "status": "SUCCESS"
    }


@mcp.tool()
def query_knowledge_graph(entity_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve all canonical facts, cross-document links, and verified citations
    for an enterprise entity in the knowledge graph.
    """
    graph = fact_layer.build_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if entity_name:
        matched_node_ids = {n["id"] for n in nodes if entity_name.lower() in n.get("label", "").lower()}
        nodes = [n for n in nodes if n["id"] in matched_node_ids]
        edges = [e for e in edges if e.get("source") in matched_node_ids or e.get("target") in matched_node_ids]

    return {
        "entity_filter": entity_name,
        "nodes": nodes,
        "edges": edges,
        "status": "SUCCESS"
    }


if __name__ == "__main__":
    if hasattr(mcp, "run"):
        mcp.run()
    else:
        print("FactForge FastMCP server initialized with tools:", list(mcp.tools.keys()))
