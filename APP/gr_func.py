"""
Gradio callback helpers. Call "setup(graph, pool)" after the graph and pool are created.
"""
import gradio as gr
from rag_pipeline.memory import create_new_thread, get_chat_history, clear_thread_memory

graph = None
pool = None
def setup(_graph, _pool):
    global graph, pool
    graph = _graph
    pool = _pool

# Gradio Callbacks

def list_chat_threads() -> list:
    """
    Retrieve all distinct chat names (or thread IDs).
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Ensure threads table exists if needed
                cur.execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, name TEXT);")
                # Return formatted choices: 'name — thread_id' when name exists, else short id
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
                thread_ids = [row[0] for row in cur.fetchall()]
                
                # For each thread_id, try to get its name from the threads table; if not present, use the id itself
                choices = []
                for id in thread_ids:
                    cur.execute("SELECT name FROM threads WHERE thread_id = %s;", (id,))
                    row = cur.fetchone()
                    if row and row[0]:
                        label = row[0]
                    else:
                        label = id
                    choices.append(label)
                return choices
    except Exception as e:
        print(f"Error listing threads: {e}")
        return []

def get_most_recent_thread_id() -> str:
    """
    Return the thread_id of the most recently updated checkpoint.
    """
    attempts = [
        "SELECT thread_id FROM checkpoints ORDER BY updated_at DESC LIMIT 1;",
        "SELECT thread_id FROM checkpoints ORDER BY created_at DESC LIMIT 1;",
        "SELECT thread_id FROM checkpoints ORDER BY id DESC LIMIT 1;",
    ]
    
    # Tries several common timestamp column names, then falls back to highest id. Returns empty string if none found.
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for sql in attempts:
                    try:
                        cur.execute(sql)
                        row = cur.fetchone()
                        if row and row[0]:
                            return row[0]
                    except Exception:
                        # column may not exist; try next
                        continue
    except Exception as e:
        print(f"Error querying most recent thread: {e}")
    return ""

def set_chat_name(thread_id: str, name: str):
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Ensure threads table exists if needed, then insert or update the chat name using thread_id
                cur.execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, name TEXT);")
                cur.execute("INSERT INTO threads(thread_id, name) VALUES (%s, %s) ON CONFLICT (thread_id) DO UPDATE SET name = EXCLUDED.name;", (thread_id, name))
    except Exception as e:
        print(f"Error setting thread name: {e}")
    
def _chatbot_rows(history: list) -> list:
    """
    Convert stored history (list of dicts with roles) to gr.Chatbot rows (list of (user, assistant) tuples).
    """
    if not history:
        return []
    
    # Walk through history and pair user/assistant messages when possible
    rows = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            user_msg = history[i]["content"]
            assistant_msg = ""
            # If next message is assistant, pair it; otherwise show user message with empty assistant response
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                assistant_msg = history[i + 1]["content"]
                i += 1
            rows.append((user_msg, assistant_msg))
        elif history[i]["role"] == "assistant":
            # If assistant appears without a preceding user, show it as an assistant-only row
            rows.append(("", history[i]["content"]))
        i += 1
    return rows
    
def _parse_thread_id(selection: str) -> str:
    """
    Resolve a selection (either chat name or thread id) to the underlying `thread_id`.
    """
    if not selection:
        return ""
    
    # If the selection matches a name in the `threads` table, return that id; otherwise assume the selection is the id itself.
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id FROM threads WHERE name = %s;", (selection,))
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]
    except Exception:
        pass
    return selection.strip()

def _display_chat_thread(thread_id: str) -> str:
    """
    Return the dropdown display value (chat name if present, else id).
    """
    name = _get_chat_name(thread_id)
    return name or thread_id

def _get_chat_name(thread_id: str) -> str:
    """
    Return the chat name for a given thread_id, or empty string if not found.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM threads WHERE thread_id = %s;", (thread_id,))
                row = cur.fetchone()
                return row[0] if row and row[0] else ""
    except Exception as e:
        print(f"Error getting thread name: {e}")
        return ""

def load_chat(selection: str):
    """
    Load chat history for a given chat name or thread_id: returns chatbot rows, thread state, and gr dropdown update.
    """
    thread_id = _parse_thread_id(selection)
    if not thread_id:
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads())
    
    history = None
    try:
        history = get_chat_history(graph, thread_id)
    except Exception as e:
        print(f"Error loading thread: {e}")
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads())

    rows = _chatbot_rows(history)
    display = _display_chat_thread(thread_id)
    
    return rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display)

def new_chat(name_input: str):
    """
    Create a new chat with an optional name: returns chatbot rows, thread state, and gr dropdown update.
    """
    thread_id = create_new_thread()
    
    # If user provided a name, save it in the threads table and use it for display; otherwise use the thread_id as display.
    if name_input:
        set_chat_name(thread_id, name_input)
        display = name_input
    else:
        display = thread_id
        
    return [], {"thread_id": thread_id, "rows": []}, gr.update(choices=list_chat_threads(), value=display)

def rename_chat(dropdown_selection: str, name_input: str):
    """
    Rename an existing chat: returns chatbot rows, thread state, and gr dropdown update.
    """
    thread_id = _parse_thread_id(dropdown_selection)
    if not thread_id:
        return gr.update(choices=list_chat_threads()) # No valid selection to rename, refresh dropdown choices
    
    set_chat_name(thread_id, name_input)
    display = name_input
    
    # After renaming, reload the chat to update the display and ensure the dropdown reflects the new name.
    return [], {"thread_id": thread_id, "rows": get_chat_history(graph, thread_id) and _chatbot_rows(get_chat_history(graph, thread_id)) or []}, gr.update(choices=list_chat_threads(), value=display)

def delete_chat(dropdown_selection: str):
    """
    Delete an existing chat and its history: returns (empty) chatbot rows, thread state, and gr dropdown update.
    """
    thread_id = _parse_thread_id(dropdown_selection)
    if not thread_id:
        return gr.update(choices=list_chat_threads()) # No valid selection to delete, refresh dropdown choices
    
    try:
        clear_thread_memory(graph, thread_id)  # Clear from graph (PostgreSQL)
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # remove checkpoints for thread
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (thread_id,))
                # remove thread metadata
                cur.execute("DELETE FROM threads WHERE thread_id = %s;", (thread_id,))
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        return gr.update(choices=list_chat_threads())

    # After deletion, return empty chat and reset dropdown to first available thread or empty if none remain.
    choices = list_chat_threads()
    value = choices[0] if choices else ""

    return [], {"thread_id": "", "rows": []}, gr.update(choices=choices, value=value)
