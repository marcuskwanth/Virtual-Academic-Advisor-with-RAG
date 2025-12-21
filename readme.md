# Virtual Academic Advisor Chatbot
Author: Marcus Kwan <br>
Type: FYP, private repository <br>
Technical information: DeepSeek-R1, BGE-m3, ChromaDB, LangChain

## PROGRESS HISTORY
2025-12-21 (2):
- Evaluated the basic LLM, 3 query-enhanced LLM, and basic LLM but with semantic routing.

2025-12-21 (1):
- Added the website URL to each chunks for PolyU SAO & CUS webpages, and the EEE programme booklet URL for programme documents.
    - Also prompted the LLM to refer to the webpages if deemed necessary.
- Removed some excluded webpages in PolyU SAO.
- Updated stripped webpage range for SAO webpage to better include the nav bar.
- **CRITICAL FIX**: Correct import for OllamaEmbeddings (must be langchain_community.embeddings, NOT langchain_ollama), otherwise, the quality of the retrieved result will be different!

2025-12-19 (1):
- Investigating the poor performance (won't retrieve sao/cus data at all? Maybe need routing)

2025-12-18 (1):
- Added evaluations, to be tested.

2025-12-17 (1):
- Re-organized folder structure.
- Modified chroma_db path to always points to that folder from the root directory!
- Added source and programme title (using LLM) for the PDF chunks.

2025-12-16 (1):
- Amended prompt with correct grammar and formatting.

2025-12-15 (1):
- Added Sub-query Implementation. (But it takeas too long to execute!)

2025-12-10/11 (1):
- Updated LLM Prompt (more specific on not saying 'context')
- Added RAG-Fusion Implementation.
- Added Step-back Implementation.

2025-11-07 (1):
- Fixed the Chroma operation that overrides documents (by adding extra ID digits for now).

2025-11-04 (1):
- Added a flag for enabling reasoning in the DeepSeek-R1 model.
- Fixed the output not including the reasoning content (in the invoking procedure).
- Amended the location of Chroma database to the root directory (in every .ipynb files).

2025-11-03 (1):
- Switched using single Chroma collection (advanced multiple-collection retrieval require function-calling, which DeepSeek-R1 does not support, need to find alternatives later).
- Implemented a simple RAG -> LLM program with LangGraph + LangSmith for procedure tracing.

2025-10-31 (1):
- Implemented text-splitting and doc-embedding, with modified metadata for debugging.
- Implemented structure for Chroma database.
- Amended PDF text-extraction library to PyMuPDFLoader instead of PyPDF2.
- Experimented with a LLM prompt.

2025-10-27 (1):
- Added a Github branch (main) in the repo.

2025-10-26 (1):
- Added webpage and PDF text-extraction functionality using Langchain libraries.
- Added code for filtering some non-relevant webpage.