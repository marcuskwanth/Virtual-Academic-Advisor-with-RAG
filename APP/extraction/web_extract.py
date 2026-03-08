"""Web extraction for PolyU SAO and CUS websites."""

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# HTML parser

def bs4_regex_enhance(html: str) -> str:
    """
    Parse raw HTML into clean text, preferring main-content areas.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else "No Title"

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="content")
    )
    if main_content:
        text = re.sub(r"\n\n+", "\n\n", main_content.get_text()).strip()
    else:
        for element in soup.find_all(["header", "footer", "nav"]):
            element.decompose()
        text = re.sub(r"\n\n+", "\n\n", soup.text).strip()

    return f"{title}\n\n{text}"

# Document loaders

def extract_sao_docs(sao_url: str = "https://www.polyu.edu.hk/sao/") -> list:
    """
    Crawl the PolyU SAO website and return a list of LangChain Documents.
    """
    if unwanted_metadata is None:
        unwanted_metadata = ["language"]

    loader = RecursiveUrlLoader(
        max_depth=6,
        url=sao_url,
        base_url=sao_url,
        prevent_outside=True,
        exclude_dirs=[
            sao_url + "News-and-Events",
            sao_url + "news-and-events",
            sao_url + "Sitemap",
            sao_url + "sitemap",
            sao_url + "Search-Result",
            sao_url + "search-result",
            sao_url + "Personal-Information-Collection-Statement",
            sao_url + "docdrive",
            sao_url + "-",
        ],
        extractor=bs4_regex_enhance,
    )

    docs = []
    for doc in loader.lazy_load():
        print(doc.metadata.get("source"))
        for key in unwanted_metadata:
            doc.metadata.pop(key, None)
        docs.append(doc)
    return docs


def extract_cus_docs(cus_url: str = "https://www.polyu.edu.hk/cus/") -> list:
    """
    Crawl the PolyU CUS website and return a list of LangChain Documents.
    """
    if unwanted_metadata is None:
        unwanted_metadata = ["language"]

    loader = RecursiveUrlLoader(
        max_depth=6,
        url=cus_url,
        base_url=cus_url,
        prevent_outside=True,
        exclude_dirs=[
            cus_url + "about-ous",
            cus_url + "about-cus",
            cus_url + "Sitemap",
            cus_url + "sitemap",
            cus_url + "Search-Result",
            cus_url + "search-result",
            cus_url + "internal",
            cus_url + "-",
        ],
        extractor=bs4_regex_enhance,
    )

    docs = []
    for doc in loader.lazy_load():
        print(doc.metadata.get("source"))
        for key in unwanted_metadata:
            doc.metadata.pop(key, None)
        docs.append(doc)
    return docs


# Text splitting

def split_web_docs(docs: list, source_label: str, chunk_size: int = 1200, chunk_overlap: int = 300) -> list:
    """
    Split web documents into chunks and enrich metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "N/A")
        m = re.search(r"/([^/]+)/?$", source)
        url_end = m.group(1) if m else "unknown"
        chunk.metadata["chunk_id"] = f"{source_label}_{url_end}_chunk_{i}"
        chunk.page_content = (
            f"--- {source_label} Website URL: {source} ---\n\n{chunk.page_content}"
        )

    print(f"Number of chunks: {len(chunks)}")
    return chunks
