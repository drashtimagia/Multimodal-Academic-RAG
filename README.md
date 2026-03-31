# 📚 Multimodal Academic Research RAG

A production-grade, full-stack Retrieval-Augmented Generation (RAG) system designed specifically for interacting with academic research papers. It ingests PDFs, extracts structured data (Text, Tables, and Images via OCR), and allows you to chat with your research using an advanced LangGraph reasoning engine powered by Groq's high-speed LLaMA 3.3.

## ✨ Key Features

- **Full Multimodality:** Understands text paragraphs, complex markdown tables, and parses diagrams/figures using Tesseract OCR.
- **Smart RAG Pipeline:** Built on **LangGraph**, it rewrites ambiguous user queries, retrieves relevant chunks, grades the retrieved context, and generates highly intelligent, naturally synthesized answers.
- **Contextual Suggestions:** Automatically generates 3 relevant follow-up questions at the end of every answer by analyzing the ongoing conversational history.
- **Modern Full-Stack UI:** A bespoke React + TypeScript Frontend featuring typewriter effects, expanding source cards, modality badges (¶ Text, ⊞ Table, ⊡ Image), and a sidebar for filtering across specific papers.
- **Async Ingestion Engine:** FastAPI backend running heavy multimodal pipeline steps in the background. Features progress tracking (`/ingest/status`) and a scheduled nightly cron job (APScheduler) for re-ingestion.

---

## 🏗️ Architecture & Pipeline

### 1. Ingestion Pipeline
When a PDF is dropped into `data/raw_pdfs/`, the background worker runs 4 sequential steps:
1. **Extraction (`extract_multimodal.py`):** Uses `pdfplumber` to extract text blocks, convert tables into rich Markdown, and crop images/figures. Skips already processed papers instantly.
2. **Vision/OCR (`caption_images_local_ocr.py`):** Uses PyTesseract to extract text from diagrams and charts.
3. **Chunking (`chunk_elements.py`):** Semantically chunks text and OCR data while preserving markdown tables whole.
4. **Vector Storage (`embed_and_store.py`):** Embeds the multimodal chunks using `all-MiniLM-L6-v2` and persists them in a local ChromaDB collection.

### 2. Retrieval & Generation Pipeline (LangGraph)
1. **Query Rewriter:** Uses chat history to resolve pronouns and optimize the user's raw question for vector search.
2. **Dense Retriever:** Queries ChromaDB specifically filtering on the user's selected PDFs.
3. **Context Grader:** (Optional self-correction loop) Validates if the retrieved chunks genuinely answer the question.
4. **Generator:** Synthesizes the final answer using Groq (`llama-3.3-70b-versatile`) and simultaneously generates 3 intelligent follow-up questions.

---

## 🛠️ Technology Stack

**Backend Engine**
- Python 3
- FastAPI (REST + Async BackgroundTasks)
- LangChain & LangGraph (Stateful RAG Graphs)
- ChromaDB (Local Vector Store)
- HuggingFace `sentence-transformers`
- PyTesseract & pdfplumber (Extraction)

**LLM Provider**
- Groq Cloud API (Ultrafast inference)
- Model: `llama-3.3-70b-versatile`

**Frontend Interface**
- React 18 + TypeScript + Vite
- Pure CSS (Bespoke styling, glassmorphism, responsive chat layout)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- Tesseract OCR installed on your OS (`brew install tesseract` on Mac or `apt-get install tesseract-ocr` on Linux).

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Start the Backend (FastAPI)
```bash
# Activate your virtual environment
source venv2/bin/activate

# Install dependencies (if you haven't)
pip install -r requirements.txt

# Run the Uvicorn server on port 8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*(Optionally, you can run this in the background using `nohup` for stability).*

### 4. Start the Frontend (Vite)
```bash
# Open a new terminal tab
cd frontend
npm install
npm run dev
```

The UI will automatically open at `http://localhost:5173`. 

### 5. Using the System
1. Drag and drop your academic PDFs into the `data/raw_pdfs/` directory.
2. Open the UI and click **"Re-index All Papers"** in the sidebar.
3. Wait for the pipeline to finish (the UI polls the API and will show a green checkmark).
4. Start chatting! You can select specific papers in the sidebar to narrow down the LLM's context window.
