"""
ingestion/embed_and_store.py  —  LangChain-powered ingestion
-------------------------------------------------------------
Reads the multimodal chunks (output of extract_multimodal + caption_images_local_ocr +
chunk_elements pipeline) and stores them in a persistent ChromaDB vector store
using LangChain's Chroma + HuggingFaceEmbeddings.

Run AFTER:
    1. python ingestion/extract_multimodal.py
    2. python ingestion/caption_images_local_ocr.py
    3. python ingestion/chunk_elements.py

Then run:
    python ingestion/embed_and_store.py

Or for simple PDF-only ingestion (bypasses multimodal pipeline):
    python ingestion/embed_and_store.py --simple
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ──────────────────────────────────────────────────────────────────────
CHUNKS_JSONL = Path("output/multimodal/chunks.jsonl")   # multimodal pipeline output
PLAIN_CHUNKS = Path("output/chunks")                    # fallback plain-text chunks
CHROMA_DIR   = Path("output/chroma_db")
COLLECTION   = "academic_rag"

# Uses ONNX via sentence-transformers — lightweight, no GPU needed
EMBED_MODEL  = "all-MiniLM-L6-v2"

BATCH_SIZE   = 50


def load_from_multimodal(jsonl_path: Path) -> list[Document]:
    """Load LangChain Documents from the multimodal chunks.jsonl pipeline output."""
    docs: list[Document] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        text = (c.get("text") or "").strip()
        if len(text) < 10:
            continue
        meta = c.get("meta", {})
        docs.append(Document(
            page_content=text,
            metadata={
                "source":       meta.get("source", "unknown"),
                "page":         meta.get("page", 0),
                "element_type": meta.get("element_type", "text"),
                "chunk_id":     c.get("id", ""),
            },
        ))
    return docs


def load_from_plain_chunks(chunks_dir: Path) -> list[Document]:
    """Fallback: load from plain .txt chunk files."""
    import re
    docs: list[Document] = []
    for txt_file in sorted(chunks_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if len(text) < 10:
            continue
        m = re.match(r"^(.+)_chunk_(\d+)$", txt_file.stem)
        source, chunk_num = (m.group(1), int(m.group(2))) if m else (txt_file.stem, 0)
        docs.append(Document(
            page_content=text,
            metadata={"source": source, "page": 0, "element_type": "text", "chunk_id": txt_file.stem},
        ))
    return docs


def main(simple: bool = False) -> None:
    # 1. Load documents
    if not simple and CHUNKS_JSONL.exists():
        print(f"📂 Loading from multimodal pipeline: {CHUNKS_JSONL}")
        docs = load_from_multimodal(CHUNKS_JSONL)
    else:
        print(f"📂 Loading from plain chunks: {PLAIN_CHUNKS}")
        docs = load_from_plain_chunks(PLAIN_CHUNKS)

    if not docs:
        raise SystemExit("❌ No documents found. Run the ingestion pipeline first.")
    print(f"   Loaded {len(docs)} documents ({sum(len(d.page_content) for d in docs):,} chars total)")

    # 2. Embedding model
    print(f"\n🧠 Loading embedding model '{EMBED_MODEL}' …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3. Store in ChromaDB (wipe + recreate for idempotent runs)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Storing in ChromaDB at '{CHROMA_DIR}' …")

    # Process in batches to avoid memory spikes
    vectorstore = None
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        ids   = [d.metadata["chunk_id"] or f"doc_{i+j}" for j, d in enumerate(batch)]
        b     = i // BATCH_SIZE + 1
        total = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                ids=ids,
                collection_name=COLLECTION,
                persist_directory=str(CHROMA_DIR),
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vectorstore.add_documents(documents=batch, ids=ids)

        print(f"   [{b}/{total}] Stored {len(batch)} docs")

    count = vectorstore._collection.count()
    print(f"\n✅  Done — {count} vectors in '{COLLECTION}' at {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true",
                        help="Use plain text chunks (skip multimodal pipeline)")
    args = parser.parse_args()
    main(simple=args.simple)
