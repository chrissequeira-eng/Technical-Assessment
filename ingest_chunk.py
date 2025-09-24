import os
import sqlite3
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_PATH = r"Z:\Genai_Projects\Internship_projects\Knowledge_Base\chunks.db"

def load_pdfs(folder_path):
    pdf_texts = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            path = os.path.join(folder_path, file_name)
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            pdf_texts[file_name] = text
    return pdf_texts

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def store_new_chunks(chunks, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            text TEXT UNIQUE
        )
    """)
    new_count = 0
    for source, chunk in chunks:
        try:
            cursor.execute("INSERT INTO chunks (source, text) VALUES (?, ?)", (source, chunk))
            new_count += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    conn.close()
    print(f"✅ Stored {new_count} new chunks")
