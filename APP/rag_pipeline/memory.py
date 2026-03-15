"""
Chat thread and history management (in LangGraph-level).
All functions that interact with the graph accept it as an explicit argument so that main.py versions (PostgreSQL) can use the same helpers.
"""
import uuid

def create_new_thread() -> str:
    """
    Generate and return a new unique thread ID.
    """
    return str(uuid.uuid4())

def get_chat_history(graph, thread_id: str) -> list:
    """
    Return the stored chat history for a given thread.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        if state and state.values.get("chat_history"):
            return state.values["chat_history"]
        return []
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def clear_thread_memory(graph, thread_id: str) -> bool:
    """
    Reset the stored chat history for a given thread to an empty list.
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        graph.update_state(config, {"chat_history": []})
        print(f"Cleared memory for thread: {thread_id}")
        return True
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return False

def get_chat_history_text(graph, thread_id: str) -> str:
    """
    Return a string of the chat history for display in Gradio.
    """
    history = get_chat_history(graph, thread_id)
    if not history:
        return f"No chat history found for thread: {thread_id}"

    formatted = f"Chat History for Thread: {thread_id}\n{'=' * 60}\n\n"
    for idx, msg in enumerate(history, 1):
        role = msg["role"].upper()
        content = msg["content"]
        formatted += f"{idx}. [{role}]\n   {content}\n\n"
    return formatted
