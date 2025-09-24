from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

CHROMA_DIR = r"Z:\Genai_Projects\Internship_projects\Knowledge_Base\chroma_db"
BATCH_SIZE = 500

def create_or_update_chroma(documents, persist_directory=CHROMA_DIR, collection_name="knowledge_base"):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedder)

    existing_ids = set()
    try:
        existing_ids = set(vectorstore.get()["ids"])
    except Exception:
        existing_ids = set()

    new_docs, new_ids = [], []
    for doc in documents:
        doc_id = str(doc.metadata["id"])
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if new_docs:
        for i in range(0, len(new_docs), BATCH_SIZE):
            batch_docs = new_docs[i:i + BATCH_SIZE]
            batch_ids = new_ids[i:i + BATCH_SIZE]
            vectorstore.add_documents(batch_docs, ids=batch_ids)
            print(f"✅ Added batch {i//BATCH_SIZE + 1} ({len(batch_docs)} docs)")
    return vectorstore
