"""
Pipeline subpackage.

Provides:
  - config: shared LLM, embeddings, ChromaDB, and ColBERT initialisation.
  - prompts: all LangChain PromptTemplates used by the pipeline.
  - pipeline: LangGraph State, node functions, and build_graph().
  - memory: chat thread and history management helpers.
"""
from .pipeline import (
    build_graph, 
    State
)
from .memory import (
    create_new_thread,
    get_chat_history,
    display_chat_history,
    clear_thread_memory,
    get_chat_history_text,
)
