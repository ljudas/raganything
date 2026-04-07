import os
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

app = FastAPI(title="RAG-Anything API")

# ── Config from environment ────────────────────────────────────────────────
LLM_BASE_URL      = os.getenv("LLM_BASE_URL",      "http://192.168.110.109:8002/v1")
LLM_MODEL         = os.getenv("LLM_MODEL",          "qwen3-distilled")
VLM_BASE_URL      = os.getenv("VLM_BASE_URL",       "http://192.168.110.109:8003/v1")
VLM_MODEL         = os.getenv("VLM_MODEL",          "qwen25-vl")
EMBEDDING_BASE_URL= os.getenv("EMBEDDING_BASE_URL", "http://192.168.110.109:8001/v1")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",    "embedding")
EMBEDDING_DIM     = int(os.getenv("EMBEDDING_DIM",  "1024"))
WORKING_DIR       = os.getenv("WORKING_DIR",        "/app/rag_storage")
OUTPUT_DIR        = os.getenv("OUTPUT_DIR",         "/app/output")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST",      "")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT",      "5432")
POSTGRES_USER     = os.getenv("POSTGRES_USER",      "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD",  "")
POSTGRES_DB       = os.getenv("POSTGRES_DB",        "lightrag")

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

rag: Optional[RAGAnything] = None


# ── Model helpers ──────────────────────────────────────────────────────────
def make_llm_func(base_url: str, model: str):
    def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key="dummy",
            base_url=base_url,
            **kwargs,
        )
    return llm_func


def make_vision_func(base_url: str, model: str, fallback):
    def vision_func(prompt, system_prompt=None, history_messages=[],
                    image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                model, "",
                system_prompt=None, history_messages=[],
                messages=messages,
                api_key="dummy", base_url=base_url, **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                model, "",
                system_prompt=None, history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }},
                    ]},
                ],
                api_key="dummy", base_url=base_url, **kwargs,
            )
        else:
            return fallback(prompt, system_prompt, history_messages, **kwargs)
    return vision_func


# ── RAG singleton ──────────────────────────────────────────────────────────
async def get_rag() -> RAGAnything:
    global rag
    if rag is not None:
        return rag

    llm_func    = make_llm_func(LLM_BASE_URL, LLM_MODEL)
    vision_func = make_vision_func(VLM_BASE_URL, VLM_MODEL, llm_func)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=EMBEDDING_MODEL,
            api_key="dummy",
            base_url=EMBEDDING_BASE_URL,
        ),
    )

    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    if POSTGRES_HOST:
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status

        lightrag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={
                "postgres_user":     POSTGRES_USER,
                "postgres_password": POSTGRES_PASSWORD,
                "postgres_host":     POSTGRES_HOST,
                "postgres_port":     int(POSTGRES_PORT),
                "postgres_database": POSTGRES_DB,
            },
        )
        await lightrag_instance.initialize_storages()
        await initialize_pipeline_status()

        rag = RAGAnything(
            config=config,
            lightrag=lightrag_instance,
            vision_model_func=vision_func,
        )
    else:
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embedding_func,
        )

    return rag


# ── Lifecycle ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await get_rag()


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload and ingest a document into the RAG knowledge base."""
    r = await get_rag()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        await r.process_document_complete(
            file_path=tmp_path,
            output_dir=OUTPUT_DIR,
            parse_method="auto",
            backend="pipeline",   # CPU-only – no GPU needed on Hetzner
        )
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    vlm_enhanced: bool = True


@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG knowledge base."""
    r = await get_rag()
    try:
        result = await r.aquery(
            request.query,
            mode=request.mode,
            vlm_enhanced=request.vlm_enhanced,
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
