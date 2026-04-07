# RAG-Anything Service

FastAPI-Wrapper für RAG-Anything, deployed auf Hetzner via Coolify.

## Endpoints

- `GET /health` – Health check
- `POST /upload` – PDF/Dokument hochladen und indexieren
- `POST /query` – RAG-Anfrage stellen

## Environment Variables

| Variable | Default | Beschreibung |
|----------|---------|--------------|
| `LLM_BASE_URL` | `http://192.168.110.109:8002/v1` | LLM Endpoint (DGX Spark) |
| `LLM_MODEL` | `qwen3-distilled` | LLM Modellname |
| `VLM_BASE_URL` | `http://192.168.110.109:8003/v1` | VLM Endpoint (DGX Spark) |
| `VLM_MODEL` | `qwen25-vl` | VLM Modellname |
| `EMBEDDING_BASE_URL` | `http://192.168.110.109:8001/v1` | Embedding Endpoint (DGX Spark) |
| `EMBEDDING_MODEL` | `embedding` | Embedding Modellname |
| `EMBEDDING_DIM` | `1024` | Embedding Dimension |
| `POSTGRES_HOST` | `` | PostgreSQL Host (LightRAG DB) |
| `POSTGRES_PORT` | `5432` | PostgreSQL Port |
| `POSTGRES_USER` | `postgres` | PostgreSQL User |
| `POSTGRES_PASSWORD` | `` | PostgreSQL Password |
| `POSTGRES_DB` | `lightrag` | PostgreSQL Datenbank |

## Deploy via Coolify

1. Repo in Coolify als **Dockerfile** Service hinzufügen
2. Port `9622` setzen
3. Domain `raganything.larsjudas.de` konfigurieren
4. Environment Variables setzen
5. Deploy
