"""
rag/graph.py  —  LangGraph Corrective RAG pipeline (multimodal-aware)
----------------------------------------------------------------------
4-node graph:
    query_rewriter → retriever → doc_grader → generator

The generator handles TEXT, TABLE, and IMAGE (OCR) chunks differently,
giving the LLM explicit instructions for each modality.
Supports optional paper filtering via state["paper_filter"].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR  = Path(__file__).parent.parent / "output" / "chroma_db"
COLLECTION  = "academic_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 6
MAX_RETRIES = 1

# ── Shared resources ──────────────────────────────────────────────────────────
_vectorstore: Optional[Chroma] = None
_llm: Optional[ChatOpenAI] = None


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
    return _vectorstore


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
    return _llm


# ── State ─────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:     str
    chat_history: List[str]
    rewritten_q:  str
    paper_filter: List[str]       # e.g. ["attention_is_all_you_need", "bert"]
    documents:    List[Document]
    graded_docs:  List[Document]
    answer:       str
    sources:      List[dict]
    suggested_questions: List[str]
    retries:      int


# ── Helpers ───────────────────────────────────────────────────────────────────
def _format_chunk_for_context(i: int, doc: Document) -> str:
    """Format a chunk for the LLM prompt with modality-specific framing."""
    meta = doc.metadata
    etype   = meta.get("element_type", "text")
    source  = meta.get("source", "unknown")
    page    = meta.get("page", "?")

    if etype == "table":
        label = f"[Chunk {i+1} | TABLE | {source} p.{page}]"
        note  = "(This is structured tabular data in Markdown format.)"
    elif etype == "image":
        label = f"[Chunk {i+1} | IMAGE (OCR) | {source} p.{page}]"
        note  = "(This is text extracted via OCR from a figure/diagram.)"
    else:
        label = f"[Chunk {i+1} | TEXT | {source} p.{page}]"
        note  = ""

    return f"{label} {note}\n{doc.page_content}"


# ── Nodes ─────────────────────────────────────────────────────────────────────
def query_rewriter(state: RAGState) -> RAGState:
    llm = _get_llm()
    history = "\n".join(state.get("chat_history", []))
    resp = llm.invoke([
        SystemMessage(content=(
            "You are a query rewriting expert for academic research papers. "
            "Rewrite the user's question to be more specific and retrieval-friendly "
            "for searching across research paper text, tables, and figures. "
            "If relevant, take their chat history into account to resolve pronouns. "
            "Output ONLY the rewritten question."
        )),
        HumanMessage(content=f"HISTORY:\n{history}\n\nQUESTION: {state['question']}"),
    ])
    return {**state, "rewritten_q": resp.content.strip()}


def retriever(state: RAGState) -> RAGState:
    vs    = _get_vectorstore()
    query = state.get("rewritten_q") or state["question"]

    # Apply paper filter if specified
    where = None
    papers = [p for p in state.get("paper_filter", []) if p]
    if len(papers) == 1:
        where = {"source": papers[0]}
    elif len(papers) > 1:
        where = {"source": {"$in": papers}}

    if where:
        docs = vs.similarity_search(query, k=TOP_K, filter=where)
    else:
        docs = vs.similarity_search(query, k=TOP_K)

    return {**state, "documents": docs}


def doc_grader(state: RAGState) -> RAGState:
    llm      = _get_llm()
    question = state["question"]
    graded: List[Document] = []

    for doc in state["documents"]:
        etype = doc.metadata.get("element_type", "text")
        resp  = llm.invoke([
            SystemMessage(content=(
                "You are a relevance grader for academic RAG. "
                f"The document is a {etype.upper()} chunk from a research paper. "
                "Reply with a single word: 'yes' if relevant to the question, 'no' if not."
            )),
            HumanMessage(content=f"Question: {question}\n\nChunk:\n{doc.page_content[:600]}"),
        ])
        if resp.content.strip().lower().startswith("yes"):
            graded.append(doc)

    retries = state.get("retries", 0)
    if not graded and retries < MAX_RETRIES:
        return {**state, "graded_docs": [], "rewritten_q": "", "retries": retries + 1}

    return {**state, "graded_docs": graded if graded else state["documents"]}


def generator(state: RAGState) -> RAGState:
    llm  = _get_llm()
    docs = state["graded_docs"]

    context = "\n\n---\n\n".join(
        _format_chunk_for_context(i, doc) for i, doc in enumerate(docs)
    )

    resp = llm.invoke([
        SystemMessage(content=(
            "You are an expert research assistant answering questions from academic papers. "
            "You MUST accurately answer ANY question across TEXT, TABLES, and FIGURES precisely as given in the paper from start to end. "
            "Answer using ONLY the provided context to write a highly intelligent, comprehensive, and natural-sounding response. "
            "Synthesize the information seamlessly. "
            "Do NOT mention internal terms like 'Chunk', 'Document', or the structure of this prompt. "
            "If citing facts, refer naturally to the paper's name or its authors.\n\n"
            "CRITICAL: At the very end of your response, on a new line, you MUST provide 3 insightful follow-up questions "
            "that the user has not asked yet, based on the context. Format exactly as this array:\n"
            "SUGGESTED_QUESTIONS: [\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
        )),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {state['question']}\n\nANSWER:"),
    ])

    import ast
    import re

    raw_answer = resp.content.strip()
    suggested = []
    
    match = re.search(r"SUGGESTED_QUESTIONS:\s*(\[.*?\])", raw_answer, re.DOTALL)
    if match:
        try:
            suggested = ast.literal_eval(match.group(1))
            raw_answer = raw_answer[:match.start()].strip()
        except Exception:
            pass

    sources = [
        {
            "chunk_id":     doc.metadata.get("chunk_id", f"chunk_{i}"),
            "source":       doc.metadata.get("source", "unknown"),
            "pdf_file":     doc.metadata.get("pdf_file", ""),
            "page":         doc.metadata.get("page", 0),
            "element_type": doc.metadata.get("element_type", "text"),
            "excerpt":      doc.page_content[:300],
        }
        for i, doc in enumerate(docs)
    ]

    return {**state, "answer": raw_answer, "sources": sources, "suggested_questions": suggested[:3]}


# ── Router ────────────────────────────────────────────────────────────────────
def should_retry(state: RAGState) -> str:
    if not state.get("graded_docs") and state.get("retries", 0) <= MAX_RETRIES:
        return "retry"
    return "generate"


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("query_rewriter", query_rewriter)
    g.add_node("retriever",      retriever)
    g.add_node("doc_grader",     doc_grader)
    g.add_node("generator",      generator)

    g.set_entry_point("query_rewriter")
    g.add_edge("query_rewriter", "retriever")
    g.add_edge("retriever",      "doc_grader")
    g.add_conditional_edges("doc_grader", should_retry, {"retry": "retriever", "generate": "generator"})
    g.add_edge("generator", END)
    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({
        "question": "What are the main evaluation metrics used?",
        "rewritten_q": "", "paper_filter": [],
        "documents": [], "graded_docs": [],
        "answer": "", "sources": [], "retries": 0,
    })
    print("Answer:", result["answer"])
    for s in result["sources"]:
        print(f"  [{s['element_type']}] {s['source']} p.{s['page']}: {s['excerpt'][:80]}…")
