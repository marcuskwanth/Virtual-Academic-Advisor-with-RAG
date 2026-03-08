"""ChromaDB initialization and management."""

import pathlib
from chromadb.config import Settings
from chromadb import Client, PersistentClient

def get_root_dir() -> pathlib.Path:
    """
    Climbing up the directory tree to locate the root directory.
    """
    current_file_dir = pathlib.Path(pathlib.Path.cwd()).resolve().parent
    if (current_file_dir / ".git").exists():
        return current_file_dir
    for parent in current_file_dir.parents:
        if (parent / ".git").exists():
            return parent
    return current_file_dir

def get_chroma_client(chroma_db_path: str = None) -> PersistentClient:
    """
    Return a persistent ChromaDB client.
    """
    if chroma_db_path is None:
        chroma_db_path = str(get_root_dir() / "chroma_db")
    Client(Settings())  # initialise ephemeral client
    return PersistentClient(path=chroma_db_path)

def create_collection(client: PersistentClient, name: str):
    return client.create_collection(name=name)

def delete_collection(client: PersistentClient, name: str):
    client.delete_collection(name=name)

def get_collection_count(client: PersistentClient, name: str) -> int:
    return client.get_collection(name=name).count()

def list_collections(client: PersistentClient):
    return client.list_collections()

def test_vector_store(vector_store, query: str, k: int = 7) -> list:
    """
    Run a similarity search and print results for testing.
    """
    results = vector_store.similarity_search(query, k=k)
    for result in results:
        print("=" * 60)
        print(f"Content: {result.page_content}")
        print(f"Source:  {result.metadata.get('source')}")
        print(f"Chunk ID: {result.metadata.get('chunk_id')}\n")
    return results