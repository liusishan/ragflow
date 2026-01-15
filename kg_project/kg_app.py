from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
from fastapi import FastAPI, HTTPException
from networkx.readwrite import json_graph
from pydantic import BaseModel, Field

from kg_core.graph_extractor_general import GraphExtractor as GeneralGraphExtractor
from kg_core.graph_extractor_light import GraphExtractor as LightGraphExtractor
from kg_core.llm_client import LLMConfig as CoreLLMConfig, OpenAICompatibleChat
from kg_core.token_utils import num_tokens_from_string


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(title="KG Extractor", lifespan=lifespan)


class LLMConfig(BaseModel):
    model: str = Field(..., description="LLM model name")
    api_key: str = Field(..., description="LLM API key")
    base_url: str | None = Field(default=None, description="LLM API base URL")
    max_tokens: int = Field(default=8192, description="Max tokens for the model")
    temperature: float = Field(default=0.2, description="Sampling temperature")


class ChunkingConfig(BaseModel):
    max_tokens: int = Field(default=800, description="Max tokens per chunk")
    overlap_tokens: int = Field(default=0, description="Approximate overlap tokens between chunks")


class GraphExtractRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw input text")
    chunks: list[str] | None = Field(default=None, description="Pre-split chunks")
    llm: LLMConfig
    method: str = Field(default="light", description="Extraction method: light|general")
    language: str = Field(default="Chinese", description="Language used in prompts")
    entity_types: list[str] | None = Field(default=None, description="Entity types to extract")
    chunking: ChunkingConfig | None = None
    limit_nodes: int = Field(default=256, description="Max nodes in output")
    limit_edges: int = Field(default=128, description="Max edges in output")


class GraphQueryRequest(BaseModel):
    graph: dict[str, Any]
    entity_ids: list[str] = Field(default_factory=list)
    include_neighbors: bool = True
    depth: int = 1
    limit_nodes: int = 256
    limit_edges: int = 128


def _build_llm(config: LLMConfig):
    core_config = CoreLLMConfig(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    return OpenAICompatibleChat(core_config)


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    approx_chars = max(max_tokens * 4, 200)
    return [text[i : i + approx_chars].strip() for i in range(0, len(text), approx_chars) if text[i : i + approx_chars].strip()]


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    approx_chars = max(overlap_tokens * 4, 0)
    if approx_chars <= 0:
        return ""
    return text[-approx_chars:]


def _split_into_chunks(text: str, chunking: ChunkingConfig | None) -> list[str]:
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not chunking:
        return paragraphs

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = num_tokens_from_string(paragraph)
        if para_tokens > chunking.max_tokens:
            for piece in _split_long_text(paragraph, chunking.max_tokens):
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.append(piece)
            continue

        if current and current_tokens + para_tokens > chunking.max_tokens:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            if chunking.overlap_tokens > 0:
                overlap_text = _tail_overlap(chunk_text, chunking.overlap_tokens)
                current = [overlap_text] if overlap_text else []
                current_tokens = num_tokens_from_string(overlap_text) if overlap_text else 0
            else:
                current = []
                current_tokens = 0

        current.append(paragraph)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _build_graph(nodes: list[dict], edges: list[dict]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        name = node.get("entity_name")
        if not name:
            continue
        attrs = {k: v for k, v in node.items() if k != "entity_name"}
        graph.add_node(name, **attrs)
    for edge in edges:
        src = edge.get("src_id")
        tgt = edge.get("tgt_id")
        if not src or not tgt:
            continue
        attrs = {k: v for k, v in edge.items() if k not in {"src_id", "tgt_id"}}
        graph.add_edge(src, tgt, **attrs)
    if graph.number_of_nodes() > 0:
        pagerank = nx.pagerank(graph)
        for node_name, score in pagerank.items():
            graph.nodes[node_name]["pagerank"] = score
    return graph


def _trim_graph_payload(payload: dict[str, Any], limit_nodes: int, limit_edges: int) -> dict[str, Any]:
    if "nodes" in payload:
        payload["nodes"] = sorted(payload["nodes"], key=lambda x: x.get("pagerank", 0), reverse=True)[:limit_nodes]
        if "edges" in payload:
            node_id_set = {node["id"] for node in payload["nodes"]}
            filtered_edges = [
                edge
                for edge in payload["edges"]
                if edge["source"] != edge["target"]
                and edge["source"] in node_id_set
                and edge["target"] in node_id_set
            ]
            payload["edges"] = sorted(filtered_edges, key=lambda x: x.get("weight", 0), reverse=True)[:limit_edges]
    return payload


def _subgraph_from_payload(payload: dict[str, Any], entity_ids: list[str], include_neighbors: bool, depth: int) -> nx.Graph:
    graph = json_graph.node_link_graph(payload, edges="edges")
    if not entity_ids:
        return graph
    nodes = set(entity_ids)
    if include_neighbors:
        for _ in range(max(depth, 1)):
            expanded = set(nodes)
            for node in list(nodes):
                if node in graph:
                    expanded.update(graph.neighbors(node))
            nodes = expanded
    return graph.subgraph(nodes).copy()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/graph/extract")
async def extract_graph(req: GraphExtractRequest) -> dict:
    if not req.text and not req.chunks:
        raise HTTPException(status_code=400, detail="Either 'text' or 'chunks' must be provided.")

    chunks = req.chunks or _split_into_chunks(req.text or "", req.chunking)
    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks found in input.")

    llm = _build_llm(req.llm)
    method = req.method.lower()
    if method == "general":
        extractor = GeneralGraphExtractor(llm, language=req.language, entity_types=req.entity_types)
    elif method == "light":
        extractor = LightGraphExtractor(llm, language=req.language, entity_types=req.entity_types)
    else:
        raise HTTPException(status_code=400, detail="Unsupported method. Use 'light' or 'general'.")

    nodes, edges = await extractor("input-text", chunks)
    graph = _build_graph(nodes, edges)
    payload = json_graph.node_link_data(graph, edges="edges")
    payload = _trim_graph_payload(payload, req.limit_nodes, req.limit_edges)

    return {
        "graph": payload,
        "stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "chunks": len(chunks),
        },
    }


@app.post("/graph/query")
async def query_graph(req: GraphQueryRequest) -> dict:
    graph = _subgraph_from_payload(req.graph, req.entity_ids, req.include_neighbors, req.depth)
    payload = json_graph.node_link_data(graph, edges="edges")
    payload = _trim_graph_payload(payload, req.limit_nodes, req.limit_edges)
    return {"graph": payload}


def main() -> None:
    import uvicorn

    uvicorn.run("kg_app:app", host="0.0.0.0", port=8010)
