"""
Pipeline subpackage.

Provides:
  - config:     Shared LLMs, embedding, ChromaDB, and ColBERT initialization.
  - prompts:    All LangChain PromptTemplates used by the pipeline.
  - pipeline:   LangGraph State, node functions, and build_graph().
  - memory:     Chat thread and history management helpers.
"""
from .pipeline import (
    build_graph, 
    State
)
from .memory import (
    create_new_thread,
    get_chat_history,
    clear_thread_memory,
    get_chat_history_text,
)
