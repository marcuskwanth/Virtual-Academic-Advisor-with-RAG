"""
Extraction subpackage.

Functions:
  - Web scraping (PolyU SAO and CUS websites)
  - PDF extraction
  - ChromaDB operations
"""
from .chroma_operation import (
    get_root_dir,
    get_chroma_client,
    create_collection,
    delete_collection,
    get_collection_count,
    list_collections,
    test_vector_store,
)
from .web_extract import (
    bs4_regex_enhance,
    extract_sao_docs,
    extract_cus_docs,
    split_web_docs,
    generate_embeddings,
    embed_chunks_to_chroma,
)
from .pdf_extract import (
    extract_programme_title,
    extract_pdf_docs,
    summarize_table_element,
    summarize_tables,
    save_table_docs,
    load_table_docs,
    split_and_prepare_pdf_chunks,
)
from .embedding import (
    generate_embeddings, 
    embed_chunks_to_chroma,
)
