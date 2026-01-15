# KG Extractor (Standalone Project)

This is a **standalone** knowledge-graph extraction service. Input is plain text (or pre-split chunks). File-to-text conversion happens upstream.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Setup (uv)

```bash
cd kg_project
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

## Run (Development)

```bash
uv run uvicorn kg_app:app --host 0.0.0.0 --port 8010
```

## Run (Production)

```bash
uv run uvicorn kg_app:app \
  --host 0.0.0.0 \
  --port 8010 \
  --workers 2
```

If you want a systemd service, create a unit like:

```ini
[Unit]
Description=KG Extractor Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/kg_project
ExecStart=/opt/kg_project/.venv/bin/uv run uvicorn kg_app:app --host 0.0.0.0 --port 8010 --workers 2
Restart=always

[Install]
WantedBy=multi-user.target
```

## API

### Health

```
GET /health
```

### Extract graph

```
POST /graph/extract
```

Example payload:

```json
{
  "text": "...",
  "llm": {
    "model": "gpt-4o-mini",
    "api_key": "...",
    "base_url": "https://api.openai.com/v1",
    "max_tokens": 8192,
    "temperature": 0.2
  },
  "method": "light",
  "language": "Chinese",
  "entity_types": ["organization", "person", "geo", "event", "category"],
  "chunking": {
    "max_tokens": 800,
    "overlap_tokens": 0
  }
}
```

### Query subgraph

```
POST /graph/query
```

Example payload:

```json
{
  "graph": { "nodes": [], "edges": [] },
  "entity_ids": ["ACME", "Alice"],
  "include_neighbors": true,
  "depth": 1
}
```
