"""
Shared runtime config initialization for the RAG pipeline:
  - LLM models (llm, simpler_llm) via Ollama
  - Embedding function (emb)
  - ChromaDB persistent client and LangChain VectorStore (vectorStore)
  - ColBERT reranker (colbert) — optional, controlled by USE_COLBERT

All values can be overridden via environment variables:
    - CHROMA_DB_PATH: Filesystem path for ChromaDB persistence (default: ./chroma_db)
    - USE_COLBERT:    Whether to load the ColBERT reranker (default: 1)
    - SINGLE_COLLECTION: Whether to use a single ChromaDB collection for all documents (default: true)
    - COLLECTION_NAME: Name of the ChromaDB collection to use (default: "vaa_documents")
"""

import os
import pathlib
import platform
import multiprocessing

from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import chromadb

# Project-root

def get_root_dir() -> pathlib.Path:
    """Return the root directory of APP."""
    return pathlib.Path(__file__).parent.parent.resolve()

ROOT_DIR = get_root_dir()
print(f"\nRoot dir: {ROOT_DIR}\n")

# Variables config

CHROMA_DB_PATH      = os.environ.get("CHROMA_DB_PATH", str(ROOT_DIR / "chroma_db"))
OLLAMA_BASE_URL     = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
USE_COLBERT         = int(os.environ.get("USE_COLBERT", 1)) == 1
SINGLE_COLLECTION   = os.environ.get("SINGLE_COLLECTION", "true").lower() == "true"

COLLECTION_NAME     = "vaa_documents" if SINGLE_COLLECTION else "academic_documents"

num_queries: int    = 4       # number of alternative queries for RAG-Fusion
num_docs: int       = 8       # number of top documents to retain after reranking
num_chat_his: int   = 5       # number of past message pairs included in context

# LLMs

print(f"\nLLM: ChatOllama with model 'deepseek-r1:8b' at {OLLAMA_BASE_URL}\n")
llm = ChatOllama(model="deepseek-r1:8b", validate_model_on_init=True, temperature=0.5, reasoning=True, base_url=OLLAMA_BASE_URL,)
simpler_llm = ChatOllama(model="deepseek-r1:8b", validate_model_on_init=True, reasoning=False, base_url=OLLAMA_BASE_URL,)
emb = OllamaEmbeddings(model="bge-m3:567m", base_url=OLLAMA_BASE_URL,)

# ChromaDB & Vector Store
print(f"\nChromaDB path at: {CHROMA_DB_PATH}\n")

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
vectorStore = Chroma(
    collection_name=COLLECTION_NAME,
    client=_chroma_client,
    embedding_function=emb,
)

print(f"\nChromaDB: number of documents in collection '{COLLECTION_NAME}': {_chroma_client.get_collection(COLLECTION_NAME).count()}\n")

# BM25 retriever
_collection = _chroma_client.get_collection(COLLECTION_NAME)
docs = _collection.get(
    include=["documents", "metadatas"],
    limit=_collection.count()
)

docs_for_bm25 = [
    Document(page_content=doc_text, metadata=md)
    for doc_text, md in zip(docs["documents"], docs["metadatas"])
]

bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
print(f"BM25 retriever: initialized over {len(docs_for_bm25)} documents")

# ColBERT

colbert = None
print(f"\nColBERT enabled: {USE_COLBERT}\n")

if USE_COLBERT:
    print(f"Processor: {platform.processor()}")
    print(f"Machine:   {platform.machine()}")

    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set — safe to ignore

    colbert_gpu = os.environ.get("COLBERT_GPU")
    if colbert_gpu:
        print(f"Using ColBERT on GPU: {colbert_gpu}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = colbert_gpu
    else:
        print("Using ColBERT on CPU...")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from ragatouille import RAGPretrainedModel

    print("Loading ColBERT model...")
    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    print("ColBERT model loaded")
else:
    print("ColBERT disabled")
