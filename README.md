Industrial Safety QA System

This project is a FastAPI-based Question Answering (QA) system built around an industrial safety knowledge base. The pipeline ingests PDF documents, splits them into smaller text chunks (split_text), stores them in SQLite (store_new_chunks), and embeds them into a Chroma vector database (create_or_update_chroma). Queries are handled through a /ask API endpoint: it retrieves relevant chunks using baseline search (search_with_threshold) and optionally applies a reranker (rerank) to improve ranking. Supporting utilities include load_pdfs (PDF extraction), load_chunks_from_sqlite (reload stored chunks), and load_sources_json (map sources to URLs for citations).

The system supports two retrieval modes: baseline (vector similarity search only) and hybrid (baseline retrieval plus reranking for higher relevance). By default, the /ask endpoint returns retrieved text and citations directly. To enable the LLM, uncomment the prompt and llm.invoke() section in main.py; this allows the assistant to generate a natural-language answer grounded in the retrieved context. Baseline mode is faster and lightweight, while hybrid mode is more accurate for nuanced queries at the cost of extra computation.

Installation & Setup

1. Clone the repository:
   git clone <your-repo-url>
   cd <repo-folder>

2. Create and activate a Python virtual environment:
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Set environment variables in myenv/.env (e.g., for LLM API keys).

5. Place your PDFs inside the Knowledge Base folder:
   Knowledge_Base/
       file1.pdf
       file2.pdf
       sources.json

Running the API

Start the FastAPI server:

   uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000.

Example Usage

Send a POST request to /ask:

   curl -X POST "http://127.0.0.1:8000/ask" \
        -H "Content-Type: application/json" \
        -d '{"q": "Question", "k": 5, "mode": "hybrid"}'

Response:

{
  "answer": "Retrieved chunks text from the knowledge base...",
  "citations": ["source1.pdf", "source2.pdf"]
}

- mode: "baseline" → retrieves chunks using vector similarity search only.
- mode: "hybrid" → retrieves chunks and reranks them for higher relevance.
- To use the LLM for natural language answers, uncomment the llm.invoke() section in main.py.
