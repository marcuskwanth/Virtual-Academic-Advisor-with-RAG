"""
Shared runtime config initialization for the RAG pipeline:
  - LangSmith tracing environment variables
  - LLM models (llm, simpler_llm) via Ollama
  - Embedding function (emb)
  - ChromaDB persistent client and LangChain VectorStore (vectorStore)
  - ColBERT reranker (colbert) — optional, controlled by USE_COLBERT

All values can be overridden via environment variables:
    - CHROMA_DB_PATH: Filesystem path for ChromaDB persistence (default: ./chroma_db)
    - USE_COLBERT:    Whether to load the ColBERT reranker (default: true)
    - SINGLE_COLLECTION: Whether to use a single ChromaDB collection for all documents (default: true)
    - COLLECTION_NAME: Name of the ChromaDB collection to use (default: "vaa_documents")
"""

import os
import pathlib
import platform
import multiprocessing
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import chromadb

# Project-root

def get_root_dir() -> pathlib.Path:
    """Return the root by searching for a .git directory."""
    current_file_dir = pathlib.Path(pathlib.Path.cwd()).resolve().parent
    if (current_file_dir / ".git").exists():
        return current_file_dir
    for parent in current_file_dir.parents:
        if (parent / ".git").exists():
            return parent
    return current_file_dir  # fallback

ROOT_DIR = get_root_dir()

# Variables config

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", str(ROOT_DIR / "chroma_db"))
USE_COLBERT = os.environ.get("USE_COLBERT", "true").lower() == "true"
SINGLE_COLLECTION = os.environ.get("SINGLE_COLLECTION", "true").lower() == "true"
COLLECTION_NAME = "vaa_documents" if SINGLE_COLLECTION else "academic_documents"

num_queries: int = 4        # number of alternative queries for RAG-Fusion
num_docs: int = 7           # number of top documents to retain after reranking
num_chat_his: int = 3       # number of past message pairs included in context

# LangSmith tracing

date = datetime.today().strftime("%Y-%m-%d")
proj_name = f"[{date}] VAA - Gradio GUI ({'ColBERT' if USE_COLBERT else 'RAG Fusion'} + Pipeline)"

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_API_KEY","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",)
os.environ.setdefault("LANGSMITH_PROJECT", proj_name)

# LLMs

llm = ChatOllama(model="deepseek-r1:8b", validate_model_on_init=True, temperature=0.5, reasoning=True)
simpler_llm = ChatOllama(model="deepseek-r1:8b", validate_model_on_init=True, reasoning=False)
emb = OllamaEmbeddings(model="bge-m3:567m")

# ChromaDB & Vector Store

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
vectorStore = Chroma(
    collection_name=COLLECTION_NAME,
    client=_chroma_client,
    embedding_function=emb,
)

# ColBERT

colbert = None
if USE_COLBERT:
    print(f"Processor: {platform.processor()}")
    print(f"Machine:   {platform.machine()}")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set — safe to ignore

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from ragatouille import RAGPretrainedModel

    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    print("ColBERT model loaded")
else:
    print("ColBERT disabled")
