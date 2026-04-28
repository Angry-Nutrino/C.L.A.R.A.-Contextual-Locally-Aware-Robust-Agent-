"""
rag_db_builder.py
Multi-format knowledge base builder for CLARA's consult_archive tool.
Indexes: core_logic/docs/ (PDF/MD/TXT/PY) + CLAUDE.md + ROADMAP.md

Full rebuild every time — incremental FAISS updates are fragile at this scale.
Called at startup and on source file change (via rag_rebuild background trigger).

A threading.Lock prevents concurrent rebuilds (startup + EnvironmentWatcher
can race at launch — the lock ensures only one build runs at a time).
"""
import pathlib
import threading
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .session_logger import slog

CURRENT_DIR  = pathlib.Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
DOCS_DIR     = CURRENT_DIR / "docs"
DB_PATH      = str(CURRENT_DIR / "knowledge_base")

# Files always included regardless of docs/ contents
ALWAYS_INCLUDE = [
    PROJECT_ROOT / "CLAUDE.md",
    PROJECT_ROOT / "briefs" / "ROADMAP.md",
]

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".py"}

# Prevents concurrent rebuilds (startup thread + EnvironmentWatcher race at launch)
_rebuild_lock = threading.Lock()


def _load_file(path: pathlib.Path):
    """Load a single file and return list of Document objects."""
    try:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_name"] = path.name
        return docs
    except Exception as e:
        slog.warning(f"   [RAG] Warning: Failed to load {path.name}: {e}")
        return []


def build_knowledge_base() -> str:
    """
    Full rebuild of the FAISS knowledge base.
    Serialized by _rebuild_lock — concurrent callers wait rather than double-build.
    Returns status string.
    """
    if not _rebuild_lock.acquire(blocking=False):
        slog.info("   [RAG] Rebuild already in progress — skipping duplicate call.")
        return "Rebuild already in progress."

    try:
        all_docs = []

        # 1. Load all supported files from docs/
        if DOCS_DIR.exists():
            for f in DOCS_DIR.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    docs = _load_file(f)
                    all_docs.extend(docs)
                    slog.info(f"   [RAG] Loaded: {f.name} ({len(docs)} chunks raw)")

        # 2. Always include CLAUDE.md and ROADMAP.md
        for fixed_path in ALWAYS_INCLUDE:
            if fixed_path.exists():
                docs = _load_file(fixed_path)
                all_docs.extend(docs)
                slog.info(f"   [RAG] Loaded: {fixed_path.name} ({len(docs)} chunks raw)")

        if not all_docs:
            slog.warning("   [RAG] No source documents found. Knowledge base not built.")
            return "No source documents found."

        # 3. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        chunks = splitter.split_documents(all_docs)
        slog.info(f"   [RAG] Total chunks after splitting: {len(chunks)}")

        # 4. Build embeddings and save
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(DB_PATH)

        msg = f"Knowledge base rebuilt: {len(chunks)} chunks from {len(all_docs)} raw docs."
        slog.info(f"   [RAG] {msg}")
        return msg

    finally:
        _rebuild_lock.release()


if __name__ == "__main__":
    build_knowledge_base()
