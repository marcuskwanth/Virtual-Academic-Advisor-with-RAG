# PolyU Virtual Academic Advisor Chatbot
Author: Marcus Kwan <br>
Type: 2526-AIIE-FYP, private repository <br>
Technical Stacks: DeepSeek-R1 (LLM), BGE-m3 (EM), ChromaDB (RAG), LangChain (Pipeline)

## PROGRESS HISTORY
2026-3-1 (2):
- Updated webpage embedding handling:
    - Instead of hard-coding splitting position, now uses BeautifulSoups functions (title and find) to get the webpage title and main content directly. Removed unwanted metadatas.

2026-3-1 (1):
- Amended PDF processing part after testing:
    - Changed the LLM for title extraction and table summarization: DeepSeek_R1:8b -> Llama3.1:8b, for faster processing.
    - Advised LLM to include a programme title short form, instead of just the programme title in the chunks (To improves retrieval accuarcy for programme-specific questions).
    - Added infer_table_structure and hi-res strategy for Unstructured.io function (Uses extra dependencies for DNN OCR extraction, should have better table detection).
    - Removed excess metadata for PDF chunks (languages, coordinates), to prevent ChromaDB error.
    - **Critical**: Store the table documents locally (using Pickle), and use deepcopy() to load them into loop function to avoid mutation!!
- Changed RecursiveCharacterSplit parameters for raw datas: Chunk size -> 1200, overlapping size -> 300.
- Updated the prompt for question reformulating slightly.

2026-2-26/27 (1):
- Changed programme booklet PDF embedding approach for tables: RecursiveCharacterSplit -> Unstructured
    - Extract text and table parts, and process them separately. 
        - Table: Summarize them into natural language w/ LLM, embed it. Store the original table (in markup format) into the corresponding metadata. 
        - Text: Standard RCS approach for now.
- Duplicated GRADIO notebook file for the following changes:
    - Modified it to use PostgresSQL with LangChain PostgresStore() and PostgresSaver().
    - **WIP**: Modified the Gradio Interface to select chat history separately in a tab (still have bugs atm).
- Renamed files for Code_Extraction.

2026-2-14 (1):
- Added routing to the Gradio code.
    - Instead of routing to different DB, it routes based on the user input to see if its relevant to the domain.
    - If not, route to a separate function with a different LLM prompt.
- Fix: Previously used captialize character for the category result, but have wrong result... Now no need to process the category character!

2026-2-13 (1):
- Prepared 3 sample questions pairs for each test categories (in evaluation).
- Added ColBERT retrieval mechanism to the Gradio code, with a switch to change between RAG-Fusion and ColBERT.
- Added retrieval score (distance score) to the Gradio code.
- Revised LLM prompt: Now specifies the chatbot from "Department of EEE at the PolyU", instead of just PolyU.
- Changed the temperature of the main LLM to be 0.5 from 0.6.

2026-2-6 (1):
- Added a gradio 5 test GUI for the application. (with gr.ChatInterface())
    - Now uses in-memory approach, later see if I can port PostgresSQL into it.
    - The code follows the official Gradio guide on streaming output. 
        - First, demo in the code.
        - Modifications are needed in the graph to use llm.stream() function.
    - The chat-history saving is also moved to the stage after streaming is complete.
    - **CRITICAL FIX**: the chat-history cannot be saved correctly after modification. Need to source from GPT + official Gradio page on how to achieve! = input + output.

2026-1-31 (1):
- Fixed some connection issue in the code for PostgresSQL (mainly setup()).
- Prompt enhancement: to make sure alt. query can focus on each sub-question if required.

2026-1-30 (1):
- Implemented persistence memory for chat history in separate notebook.
    - using PostgresSQL, with LangChain PostgresStore() and PostgresSaver(), babystep thing only, need to think how to implement it in production...
    - need to set-up a local database manually using brew + commands (vaa_chat_mem_db).
- Prompt enhancement: Fixed formatting, particularly indentation, also prompt strictly no.# of alt. questions during RAG-Fusion.

2026-1-29 (1):
- Implemented stateful retrieval in chat-history program with Query Transformation.
    - By using LLM and a prompt to rewrite/contextualize the user prompt if prev. conversations are found, to reduce ambiguity!
    - Added readmes for Query Transformation.
- Prompt enhancement: ordering + never fabricate non-existence URLs!
    - After testing, still, it sometimes give non-existence URLs... (e.g. hkie.org without .hk)
    - May try to tune the temperature of the LLM instead.

2026-1-23 (1):
- Experimented ColBERT doc-ranking and its evaluation (Using Ragatouille, with reference from a classmate).
    - Thought of RankGPT as well, but concerned about the query time...
- Prompt enhancement: Updated the content ordering in the prompt.

2026-1-22 (1):
- Renamed the notebook files w/ "TYPENUMBER_NUMBER_xxx".
- Implemented in-memory Chat History persistence using LangChain InMemoryStore() (using RAG-Fusion as example).
    - With a enhanced prompt tailor-made for chat-history handling.
    - Current issue: Retriever is stateless, leading to contextual drift (i.e. follow-up question may not be contextualized with existing chat data).

2026-1-16 (1):
- Re-build the Chroma vector databases with webpages that have a max-depth of 5 instead of 2.
    - Initially has issue due to error in parasing HTML when higher depth is used. Fixed by using different parser for the HTML syntax (html.parser instead of lxml).
- Changed RAG-Fusion to now use top 5 documents in the list (a conclusion from interim).
- Changed RAG-Fusion and Step-back prompting to use non-reasoning LLM for queries generation (a conclusion from interim).
- Prompt enhancement: Advise the LLM not to fabricate URLs (discovered in RAG-Fusion testing).

## INTERIM STAGE: PROGRESS HISTORY

2025-12-26 (1):
- Added hallucination counts in he additional evaluation.

2025-12-24 (1):
- Added additional evaluation (overall results of every new modification).
- Simplified the evaluations (reduced keyword coverage).

2025-12-23 (1):
- Fixed RAG-Fusion, should not include the original question.
- Re-run the evaluation with corrected test data!

2025-12-22 (1):
- Added more visualization to the evaluated data. All methods are evaluated again!
- Added an option to load previously saved evaluation results for all notebook files.
- Hotfix: Always points to that folder from the root directory

2025-12-21 (3):
- Evaluated the basic LLM, 3 query-enhanced LLM, and basic LLM but with semantic routing.

2025-12-21 (2):
- Implemented semantic routing (using LLM without reasoning to decide which DB to use, due to speed issue).

2025-12-21 (1):
- Added the website URL to each chunks for PolyU SAO & CUS webpages, and the EEE programme booklet URL for programme documents.
    - Also prompted the LLM to refer to the webpages if deemed necessary.
- Removed some excluded webpages in PolyU SAO.
- Updated stripped webpage range for SAO webpage to better include the nav bar.
- **CRITICAL FIX**: Correct import for OllamaEmbeddings (must be langchain_community.embeddings, NOT langchain_ollama), otherwise, the quality of the retrieved result will be different!

2025-12-19 (1):
- Added visualizations to the evaluated data.
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

2025-10-20 (1):
- Initial Commit.