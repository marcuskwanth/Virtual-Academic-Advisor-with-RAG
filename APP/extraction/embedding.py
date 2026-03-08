"""Embedding helpers"""
from concurrent.futures import ThreadPoolExecutor

def generate_embeddings(chunks: list, embedding_function) -> list:
    """
    Generate embeddings for a list of document chunks.
    """
    def _embed_one(chunk):
        return embedding_function.embed_query(chunk.page_content)

    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(_embed_one, chunks))
    return embeddings


def embed_chunks_to_chroma(chunks: list, embeddings: list, collection, id_offset: int = 0) -> None:
    """
    Add document chunks and their pre-computed embeddings to a ChromaDB collection.
    E.g. of id_offset: for sao: 0, cus: 3000, pdf: 6000
    """
    for i, chunk in enumerate(chunks):
        print(
            f"Adding chunk {i + 1}/{len(chunks)} to ChromaDB."
            f" Metadata: {chunk.metadata}"
        )
        collection.add(
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
            embeddings=[embeddings[i]],
            ids=[str(i + id_offset)],
        )
    print(f"Added {len(chunks)} chunks into ChromaDB")