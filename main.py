import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from Internship_projects.ingest_chunk import load_pdfs, split_text, store_new_chunks
from Internship_projects.Embed_Chunk import create_or_update_chroma
from Internship_projects.Baseline_Search import search_with_threshold
from Internship_projects.Reranker import rerank
from langchain.docstore.document import Document
import sqlite3
from dotenv import load_dotenv

load_dotenv("myenv/.env")

KB_FOLDER = r"Z:\Genai_Projects\Internship_projects\Knowledge_Base"
DB_PATH = os.path.join(KB_FOLDER, "chunks.db")

#  Load sources.json
def load_sources_json(kb_folder):
    sources_path = os.path.join(kb_folder, "sources.json")
    if not os.path.exists(sources_path):
        return {}
    with open(sources_path, "r", encoding="utf-8") as f:
        sources_data = json.load(f)
    mapping = {}
    for entry in sources_data:
        title = entry.get("title", "Unknown Title")
        url = entry.get("url", "")
        mapping[title] = {"title": title, "url": url}
    return mapping

#  Load chunks
def load_chunks_from_sqlite(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, source, text FROM chunks")
    rows = cursor.fetchall()
    conn.close()
    documents = []
    for row in rows:
        chunk_id, source, text = row
        documents.append(Document(page_content=text, metadata={"id": chunk_id, "source": source}))
    return documents

# FastAPI 
app = FastAPI(title="Industrial Safety QA API")

class QueryRequest(BaseModel):
    q: str
    k: int = 5
    mode: str = "baseline" 
print("📦 Loading chunks and embeddings...")
pdf_data = load_pdfs(KB_FOLDER)
all_chunks = []
for filename, text in pdf_data.items():
    chunks = split_text(text)
    for chunk in chunks:
        all_chunks.append((filename, chunk))

store_new_chunks(all_chunks)
documents = load_chunks_from_sqlite()
vectorstore = create_or_update_chroma(documents)
sources_dict = load_sources_json(KB_FOLDER)
print("✅ Initialization complete.")

@app.post("/ask")
def ask(req: QueryRequest):
    try:
        results_dict = search_with_threshold(vectorstore, sources_dict, req.q, top_k=req.k)
        contexts = results_dict.get("contexts", [])

        if req.mode == "hybrid":
            contexts = rerank(contexts, req.q, method="hybrid")

        answer_texts = [c["text"] for c in contexts]
        context_text = "\n".join(answer_texts)

        citations = []
        for c in contexts:
            src_title = c.get("source") or "Unknown Source"

            src_info = sources_dict.get(src_title)
            if src_info and src_info.get("url"):
                citations.append(src_info["url"])
            else:
                citations.append(src_title)

        citations = list(set(citations))

        # Un-Comment if u want to use LLM
        """
        # If you want to use LLM later, uncomment this section
        prompt = f\"\"\"
        You are an industrial safety assistant.
        Answer the question using ONLY the context provided.
        Cite the sources (filenames or URLs) clearly in your answer.

        Question: {req.q}

        Context:
        {context_text}

        Answer:
        \"\"\"

        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            clean_answer = response.content
        elif isinstance(response, str):
            clean_answer = response
        else:
            clean_answer = str(response)
        """

        return {
            "answer": context_text.strip(),
            "citations": citations
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
