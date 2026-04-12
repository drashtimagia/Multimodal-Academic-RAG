"""
api/main.py  —  FastAPI RAG backend (LangGraph + APScheduler)

Architecture:
  - POST /query           →  LangGraph corrective RAG pipeline
  - POST /ingest          →  async background ingestion (returns job_id immediately)
  - GET  /ingest/status/{job_id}  →  poll ingestion job status + live log
  - GET  /health          →  liveness + vector count
  - APScheduler           →  nightly 2 AM auto re-ingestion (configurable via env)

Start:
    cd <project_root>
    venv2/bin/uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict = {}


# ── LangGraph graph (lazy-compiled) ──────────────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        from rag.graph import build_graph
        _graph = build_graph()
    return _graph


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []    # List of previous user questions
    top_k: int = 5
    paper_filter: list = []    # e.g. ["RAG_English"] — empty means search all papers


class Source(BaseModel):
    chunk_id: str
    source: str
    page: int
    element_type: str
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    suggested_questions: list = []
    model: str


class IngestResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    started_at: Optional[str]
    finished_at: Optional[str]
    steps_done: int
    total_steps: int
    log: str


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Academic Research RAG API",
    description="LangGraph Corrective RAG + APScheduler pipeline over academic PDFs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ingestion pipeline steps ───────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("extract_multimodal",        PROJECT_ROOT / "ingestion" / "extract_multimodal.py"),
    ("caption_images_local_ocr",  PROJECT_ROOT / "ingestion" / "caption_images_local_ocr.py"),
    ("chunk_elements",            PROJECT_ROOT / "ingestion" / "chunk_elements.py"),
    ("embed_and_store",           PROJECT_ROOT / "ingestion" / "embed_and_store.py"),
]


def run_ingestion_pipeline(job_id: str) -> None:
    """Run all ingestion steps sequentially, updating job state as we go."""
    job = _jobs[job_id]
    job["status"]     = "running"
    job["started_at"] = datetime.utcnow().isoformat()
    job["log"]        = ""

    python = sys.executable

    for i, (name, script) in enumerate(PIPELINE_STEPS):
        job["log"] += f"\n▶ [{i+1}/{len(PIPELINE_STEPS)}] {name}...\n"

        r = subprocess.run(
            [python, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            cwd=str(PROJECT_ROOT),
        )
        job["log"] += r.stdout
        job["steps_done"] = i + 1

        if r.returncode != 0:
            job["status"]      = "failed"
            job["log"]        += f"\n❌ FAILED:\n{r.stderr}"
            job["finished_at"] = datetime.utcnow().isoformat()
            return

    job["status"]      = "done"
    job["finished_at"] = datetime.utcnow().isoformat()
    job["log"]        += "\n✅ Ingestion complete."


def scheduled_ingest() -> None:
    """Called by APScheduler — creates a job and runs the pipeline."""
    job_id = f"scheduled-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    _jobs[job_id] = {
        "status": "queued",
        "started_at": None,
        "finished_at": None,
        "steps_done": 0,
        "log": "Scheduled nightly ingest started.\n",
    }
    run_ingestion_pipeline(job_id)


# ── APScheduler — nightly re-ingestion ────────────────────────────────────────
scheduler = BackgroundScheduler()

@app.on_event("startup")
def startup():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Add it to .env.")

    chroma_dir = PROJECT_ROOT / "output" / "chroma_db"
    if not chroma_dir.exists():
        raise RuntimeError(
            f"ChromaDB not found at {chroma_dir}. Run ingestion/embed_and_store.py first."
        )

    get_graph()   # warm up: compile LangGraph + load embeddings

    # Schedule nightly ingest — override hour/minute via env vars
    hour   = int(os.getenv("SCHEDULE_HOUR", "2"))
    minute = int(os.getenv("SCHEDULE_MINUTE", "0"))
    scheduler.add_job(scheduled_ingest, "cron", hour=hour, minute=minute)
    scheduler.start()

    print(f"OK  LangGraph RAG pipeline ready")
    print(f"OK  APScheduler: nightly ingest at {hour:02d}:{minute:02d} UTC")


@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown(wait=False)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        from rag.graph import _get_vectorstore
        count = _get_vectorstore()._collection.count()
    except Exception:
        count = -1
    return {
        "status": "ok",
        "vectors": count,
        "version": "2.0.0",
        "scheduler": scheduler.running,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = get_graph().invoke({
            "question":     req.question.strip(),
            "chat_history": req.chat_history,
            "rewritten_q":  "",
            "paper_filter": req.paper_filter,
            "documents":    [],
            "graded_docs":  [],
            "answer":       "",
            "sources":      [],
            "suggested_questions": [],
            "retries":      0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        Source(
            chunk_id     = s.get("chunk_id", ""),
            source       = s.get("source", "unknown"),
            page         = int(s.get("page", 0)),
            element_type = s.get("element_type", "text"),
            excerpt      = s.get("excerpt", "")[:300],
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        question = req.question,
        answer   = result.get("answer", "No answer generated."),
        sources  = sources,
        suggested_questions = result.get("suggested_questions", []),
        model    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )


@app.post("/ingest", response_model=IngestResponse, status_code=202)
def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Queue the full ingestion pipeline as a background task.
    Returns a job_id immediately — poll /ingest/status/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":      "queued",
        "started_at":  None,
        "finished_at": None,
        "steps_done":  0,
        "log":         "",
    }
    background_tasks.add_task(run_ingestion_pipeline, job_id)
    return IngestResponse(
        job_id  = job_id,
        status  = "queued",
        message = f"Ingestion pipeline queued. Poll /ingest/status/{job_id} for progress.",
    )


@app.get("/ingest/status/{job_id}", response_model=JobStatus)
def ingest_status(job_id: str):
    """Poll the status of an ingestion job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatus(
        job_id      = job_id,
        status      = job["status"],
        started_at  = job.get("started_at"),
        finished_at = job.get("finished_at"),
        steps_done  = job.get("steps_done", 0),
        total_steps = len(PIPELINE_STEPS),
        log         = job.get("log", ""),
    )


@app.get("/ingest/jobs")
def list_jobs():
    """List all ingest jobs (most recent first)."""
    return {
        jid: {
            "status":      j["status"],
            "started_at":  j.get("started_at"),
            "finished_at": j.get("finished_at"),
            "steps_done":  j.get("steps_done", 0),
        }
        for jid, j in reversed(list(_jobs.items()))
    }


@app.get("/papers")
def list_papers():
    """
    List all unique papers (by source name) indexed in ChromaDB.
    Also returns a count per paper broken down by element_type.
    """
    try:
        from rag.graph import _get_vectorstore
        vs = _get_vectorstore()
        results = vs._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    papers: dict = {}
    for m in metadatas:
        src   = m.get("source", "unknown")
        etype = m.get("element_type", "text")
        if src not in papers:
            papers[src] = {"source": src, "total": 0, "text": 0, "table": 0, "image": 0}
        papers[src]["total"] += 1
        papers[src][etype]   = papers[src].get(etype, 0) + 1

    return {
        "count": len(papers),
        "papers": sorted(papers.values(), key=lambda p: p["source"]),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF directly into the data/raw_pdfs directory."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    upload_dir = PROJECT_ROOT / "data" / "raw_pdfs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        import shutil
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": file.filename, "status": "uploaded", "message": "File uploaded successfully."}

