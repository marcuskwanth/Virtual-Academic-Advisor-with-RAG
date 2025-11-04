# Virtual Academic Advisor Chatbot
Author: Marcus Kwan
Type: FYP, Private repository

## PROGRESS HISTORY
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