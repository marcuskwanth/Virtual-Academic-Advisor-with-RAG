# Virtual Academic Advisor Chatbot
Author: Marcus Kwan
Type: FYP, private repository

## PROGRESS HISTORY
(Todo: Routing logics)

2025-12-19 (1):
- Tested evaluations in 3 different query enhancement techniques.

2025-12-17 (2):
- Added evaluations (Faithfulness, context relevancy, completeness), to be tested.

2025-12-17 (1):
- Re-organized folder structure.
- Modified chroma_db path to always points to that folder from the root directory!
- Added source and programme title for the PDF chunks.

2025-12-16 (1):
- Amended prompt with correct grammar and formatting.

2025-12-15 (1):
- Added Sub-query Implementation. (But it takeas too long to execute!)

2025-12-10 (1):
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