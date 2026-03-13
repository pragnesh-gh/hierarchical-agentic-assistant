"""Build a FAISS index from the Deep Work PDF."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import PDF_PATH, INDEX_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def build_faiss_index() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.from_documents(splits, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))


if __name__ == "__main__":
    build_faiss_index()
