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
                
                # Get all unique thread IDs from BOTH checkpoints (actual history) and threads (named but maybe no history yet)
                cur.execute("""
                    SELECT DISTINCT thread_id FROM checkpoints
                    UNION
                    SELECT DISTINCT thread_id FROM threads
                """)
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
                return sorted(list(set(choices)), key=lambda x: x.lower())
    except Exception as e:
        print(f"Error listing threads: {e}")
        return []

def get_most_recent_thread_id() -> str:
    """
    Return the thread_id of the most recently checkpoint.
    """
    attempts = [
        "SELECT thread_id FROM checkpoints ORDER BY id DESC LIMIT 1;",
    ]
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
    if not selection:
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads(), value="")

    thread_id = _parse_thread_id(selection)
    if not thread_id:
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads(), value="")
    
    history = None
    try:
        history = get_chat_history(graph, thread_id)
    except Exception as e:
        print(f"Error loading thread {thread_id}: {e}")
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads(), value="")

    rows = _chatbot_rows(history)
    display = _display_chat_thread(thread_id)
    
    return rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display)

def new_chat(name_input: str):
    """
    Create a new chat with an optional name: returns chatbot rows, thread state, and gr dropdown update.
    """
    if not name_input:
        gr.Warning("Please enter a chat title to create a new chat.")
        return gr.skip(), gr.skip(), gr.skip()
    
    # Check if name already exists in dropdown choices
    if name_input and (name_input in list_chat_threads() or name_input in [tid for tid in list_chat_threads()]):
        gr.Warning(f"Chat: '{name_input}' already exists. Please choose a different title.")
        return gr.skip(), gr.skip(), gr.skip()

    thread_id = create_new_thread()
    set_chat_name(thread_id, name_input)
    
    choices = list_chat_threads()
    
    # If the current choices is empty, make sure the new name is included in the choices returned to the dropdown
    if name_input not in choices:
        choices.append(name_input)
        
    return [], {"thread_id": thread_id, "rows": []}, gr.update(choices=choices, value=name_input)

def rename_chat(dropdown_selection: str, name_input: str):
    """
    Rename an existing chat: returns chatbot rows, thread state, and gr dropdown update.
    """
    if not dropdown_selection:
        gr.Warning("Please select a chat to rename.")
        return gr.skip(), gr.skip(), gr.skip()

    if not name_input:
        gr.Warning("Please enter a new chat title.")
        return gr.skip(), gr.skip(), gr.skip()
    
    if name_input not in list_chat_threads():
        pass

    # Check if the new name already exists in the list
    if name_input in list_chat_threads() or name_input in [tid for tid in list_chat_threads()]:
        gr.Warning(f"Chat: '{name_input}' already exists. Please choose a different title.")
        return gr.skip(), gr.skip(), gr.skip()

    thread_id = _parse_thread_id(dropdown_selection)
    if not thread_id:
        return gr.update(choices=list_chat_threads(), value="") # No valid selection to rename, refresh dropdown choices
    
    set_chat_name(thread_id, name_input)
    display = name_input
    
    # After renaming, reload the chat to update the display and ensure the dropdown reflects the new name.
    history = get_chat_history(graph, thread_id)
    rows = _chatbot_rows(history) if history else []
    return rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display)

def delete_chat(dropdown_selection: str):
    """
    Delete an existing chat and its history: returns (empty) chatbot rows, thread state, and gr dropdown update.
    """
    if not dropdown_selection:
        gr.Warning("Please select a chat to delete.")
        return gr.skip(), gr.skip(), gr.skip()
    
    if dropdown_selection not in list_chat_threads():
        pass

    thread_id = _parse_thread_id(dropdown_selection)
    if not thread_id:
        return gr.update(choices=list_chat_threads(), value="") # No valid selection to delete, refresh dropdown choices
    
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
