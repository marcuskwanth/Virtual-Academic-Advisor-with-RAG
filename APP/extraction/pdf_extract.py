"""Extraction and processing for programme PDF documents."""

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import os, re, pickle, copy

# Programme title extraction

def extract_programme_title(first_page_content: str, llm_model) -> str:
    """
    Use an LLM to extract the degree programme title from the first PDF page.
    llm_model: Any LLM wrapper with an .invoke() method (e.g., Ollama).
    """
    content_snippet = first_page_content[:2000]
    prompt = \
        f"""
        You are extracting the programme title from a university programme booklet's first page.

        *First page content*:
        {content_snippet}

        Extract the full programme title including:
        1. Degree type (e.g., Bachelor of Engineering, Master of Science, PhD)
        2. Discipline (e.g., Electrical Engineering, Electronic and Information Engineering)

        Return only the following:
        1. The programme title with normalized capitalizations
        2. The short form of such title (e.g., BEng / MSc in EE / EIE)
        Do NOT include any additional texts.

        The programme title and its short form in one line, separated by " | ":
        """
    
    # Extract programme title from first page using LLM
    try:
        response = llm_model.invoke(prompt)
        title = response.content.strip()
        title = re.sub(r"\s+", " ", title)
        title = title.replace('"', "").replace("'", "")
        if 10 < len(title) < 250:
            return title
        return "Unknown"
    except Exception as e:
        print(f"Extraction failed: {e}")
        return "Unknown"

# Document loading

def extract_pdf_docs(pdf_path: str, llm_model, unwanted_metadata: list = None) -> tuple:
    """
    Extract text and table elements from all PDFs in ``pdf_path``.
    E.g., "../RAG/ProgramBooklet"
    llm_model: Any LLM wrapper with an .invoke() method (e.g., Ollama).
    """

    if unwanted_metadata is None:
        unwanted_metadata = [
            "producer", "creator", "creationdate", "file_path",
            "format", "title", "subject", "keywords", "moddate",
            "author", "trapped", "modDate", "creationDate",
        ]

    pdfs = [f for f in os.listdir(pdf_path) if os.path.isfile(os.path.join(pdf_path, f))]
    text_docs = []
    raw_table_docs = []

    for pdf in pdfs:
        loader = PyMuPDFLoader(f"{pdf_path}/{pdf}")
        cur_pdf = loader.load()
        programme_title = ""
        if cur_pdf:
            programme_title = extract_programme_title(cur_pdf[0].page_content, llm_model)
            print(f"PDF: {pdf} -> Programme: {programme_title}")

        # Unstructured (hi-res, table structure enabled)
        loader = UnstructuredPDFLoader(
            f"{pdf_path}/{pdf}",
            mode="elements",
            strategy="hi_res",
            infer_table_structure=True,
        )
        print(f"Loading {pdf}...")
        elements = loader.load()

        for element in elements:
            element.metadata["source"] = pdf
            element.metadata["content_type"] = "pdf"
            element.metadata["programme_title"] = programme_title
            for key in unwanted_metadata:
                element.metadata.pop(key, None)

            if element.metadata.get("category") == "Table":
                raw_table_docs.append(element)
            else:
                text_docs.append(element)

    return text_docs, raw_table_docs

# Summarization for tables in PDFs

def summarize_table_element(element, llm_model):
    """
    Summarize each table element in natural language using an LLM.
    llm_model: Any LLM wrapper with an .invoke() method (e.g., Ollama).
    """
    print(f"Progress: Summarizing table from {element.metadata['source']}...")

    prompt = f"Summarize the following table in natural language in high detail:\n\n{element.page_content}"
    summary = llm_model.invoke(prompt).content.strip()
    element.metadata["table_content"] = element.page_content
    element.page_content = summary
    element.metadata["content_type"] = "table_summary"
    return element

def summarize_tables(raw_table_docs: list, llm_model) -> list:
    """
    Parallelization: Summarize all table documents using an LLM.
    llm_model: Any LLM wrapper with an .invoke() method (e.g., Ollama).
    """
    print(f"\nSummarizing {len(raw_table_docs)} tables...")
    
    with ThreadPoolExecutor() as executor:
        table_docs = list(executor.map(lambda e: summarize_table_element(e, llm_model), raw_table_docs))
    print("Table summarization complete.\n")
    return table_docs

# Save and Load table documents to avoid repeated summarization

def save_table_docs(table_docs: list, file_name: str = "table_docs.pkl") -> None:
    """
    Serialize table documents to a pickle file.
    """
    with open(file_name, "wb") as f:
        pickle.dump(table_docs, f)
    print(f"Exported {len(table_docs)} table documents to {file_name}")


def load_table_docs(file_name: str = "table_docs.pkl") -> list:
    """
    Deserialize table documents from a pickle file.
    """
    with open(file_name, "rb") as f:
        docs = pickle.load(f)
    print(f"Loaded {len(docs)} table documents from {file_name}")
    return docs

# Text splitting & chunk preparation

def split_and_prepare_pdf_chunks(text_docs: list, table_docs: list, eee_path: str, chunk_size: int = 1200, chunk_overlap: int = 300) -> list:
    """
    Split text documents into chunks and combine with table summaries.
    Deep-copies all documents before modification to avoid mutations.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(text_docs)
    all_chunks = [copy.deepcopy(c) for c in chunks] + [copy.deepcopy(d) for d in table_docs]

    for i, chunk in enumerate(all_chunks):
        source = chunk.metadata.get("source", "N/A")
        page = chunk.metadata.get("page_number", "N/A")
        programme = chunk.metadata.get("programme_title", "N/A")
        chunk_type = (
            "table" if chunk.metadata.get("content_type") == "table_summary"
            else "chunk"
        )
        chunk.metadata["chunk_id"] = f"PolyU_Doc_{source}_page_{page}_{chunk_type}_{i}"
        chunk.page_content = f"--- Source: {source}, Programme: {programme} --- \n--- Retrieved from: {eee_path} --- \n\n{chunk.page_content}"
        chunk.metadata.pop("programme_title", None)
        chunk.metadata.pop("coordinates", None)
        chunk.metadata.pop("languages", None)

    return all_chunks
