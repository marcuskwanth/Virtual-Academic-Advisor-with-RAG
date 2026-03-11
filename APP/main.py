"""
Gradio RAG chatbot with PostgreSQL (LangGraph PostgresSaver) persistence.

Prerequisites:
  1. PostgreSQL running:  brew services start postgresql
  2. Database:            createdb vaa_chat_mem_db
  3. Set DB_URI env var / ensure the default credentials match your setup.

Environment variables:
  DB_URI: PostgreSQL connection string (default: postgresql://markn:0605@localhost:5432/vaa_chat_mem_db)
"""

import os
import gradio as gr
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool

from rag_pipeline.config import llm
from rag_pipeline.pipeline import build_graph
from rag_pipeline.memory import create_new_thread, get_chat_history

# PostgreSQL connection pool

DB_URI = os.environ.get("DB_URI", "")
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
pool = ConnectionPool(conninfo=DB_URI, min_size=1, max_size=10, kwargs=connection_kwargs)

with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        print(f"Connected to PostgreSQL: {version[:50]}...")

# Build graph with PostgreSQL checkpointer and store

pg_memory = PostgresSaver(pool)
store = PostgresStore(pool)
pg_memory.setup()
store.setup()
graph = build_graph(checkpointer=pg_memory, store=store)

# Gradio Callbacks

def list_chat_threads() -> list:
    """
    Retrieve all distinct thread IDs from the checkpoints table.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Ensure threads table exists
                cur.execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, name TEXT);")
                # Return formatted choices: 'name — thread_id' when name exists, else short id
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
                ids = [row[0] for row in cur.fetchall()]
                choices = []
                for tid in ids:
                    cur.execute("SELECT name FROM threads WHERE thread_id = %s;", (tid,))
                    row = cur.fetchone()
                    if row and row[0]:
                        label = row[0]
                    else:
                        label = tid
                    choices.append(label)
                return choices
    except Exception as e:
        print(f"Error listing threads: {e}")
        return []

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
    Resolve a user-visible selection to the underlying `thread_id`.
    If the selection matches a thread name in the `threads` table, return that id;
    otherwise assume the selection is the id itself.
    """
    if not selection:
        return ""
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

def _display_for_thread(tid: str) -> str:
    """Return the dropdown display value for a thread id (name if present, else id)."""
    name = get_chat_name(tid)
    return name or tid

def get_most_recent_thread_id() -> str:
    """
    Return the thread_id of the most recently updated checkpoint.

    Tries several common timestamp column names, then falls back to highest id.
    Returns empty string if none found.
    """
    attempts = [
        "SELECT thread_id FROM checkpoints ORDER BY updated_at DESC LIMIT 1;",
        "SELECT thread_id FROM checkpoints ORDER BY created_at DESC LIMIT 1;",
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
                cur.execute("CREATE TABLE IF NOT EXISTS threads (thread_id TEXT PRIMARY KEY, name TEXT);")
                cur.execute("INSERT INTO threads(thread_id, name) VALUES (%s, %s) ON CONFLICT (thread_id) DO UPDATE SET name = EXCLUDED.name;", (thread_id, name))
    except Exception as e:
        print(f"Error setting thread name: {e}")

def get_chat_name(thread_id: str) -> str:
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM threads WHERE thread_id = %s;", (thread_id,))
                row = cur.fetchone()
                return row[0] if row and row[0] else ""
    except Exception as e:
        print(f"Error getting thread name: {e}")
        return ""

def load_chat(thread_id: str):
    tid = _parse_thread_id(thread_id)
    if not tid:
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads())
    history = None
    try:
        history = get_chat_history(graph, tid)
    except Exception as e:
        print(f"Error loading thread {tid}: {e}")
        return [], {"thread_id": "", "rows": []}, gr.update(choices=list_chat_threads())

    rows = _chatbot_rows(history)
    display = _display_for_thread(tid)
    return rows, {"thread_id": tid, "rows": rows}, gr.update(choices=list_chat_threads(), value=display)

def new_chat(name_input: str):
    tid = create_new_thread()
    if name_input:
        set_chat_name(tid, name_input)
        display = name_input
    else:
        display = tid
    return [], {"thread_id": tid, "rows": []}, gr.update(choices=list_chat_threads(), value=display)

def rename_chat(dropdown_selection: str, name_input: str):
    tid = _parse_thread_id(dropdown_selection)
    if not tid:
        return gr.update(choices=list_chat_threads())
    set_chat_name(tid, name_input)
    display = name_input
    # return updated dropdown selection and refresh chat history display with new name
    return [], {"thread_id": tid, "rows": get_chat_history(graph, tid) and _chatbot_rows(get_chat_history(graph, tid)) or []}, gr.update(choices=list_chat_threads(), value=display)

def delete_chat(dropdown_selection: str):
    tid = _parse_thread_id(dropdown_selection)
    if not tid:
        return gr.update(choices=list_chat_threads())
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # remove checkpoints for thread
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (tid,))
                # remove thread metadata
                cur.execute("DELETE FROM threads WHERE thread_id = %s;", (tid,))
    except Exception as e:
        print(f"Error deleting thread {tid}: {e}")
        return gr.update(choices=list_chat_threads())

    choices = list_chat_threads()
    value = choices[0] if choices else ""
    # After deletion, load the most recent chat (thread) if exists, else return empty chat
    return [], {"thread_id": "", "rows": []}, gr.update(choices=choices, value=value)

def stream_response(message: str, thread_state_obj: dict):
    """
    Generator that streams assistant tokens and persists history.
    `thread_state_obj` is a dict with keys `thread_id` and `rows`.
    This function yields four outputs: (chat_rows, thread_state_obj, dropdown_update, textbox_update).
    """
    thread_id = (thread_state_obj or {}).get("thread_id")
    rows = (thread_state_obj or {}).get("rows", []) or []

    # Ensure have a thread id
    if not thread_id:
        thread_id = create_new_thread()

    config = {"configurable": {"thread_id": thread_id}}

    # Add the user message and an assistant placeholder
    rows = rows + [(message, "Query received. Processing...")]
    display = _display_for_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display), gr.update(value="")

    # Run pipeline to prepare the prompt
    status = None
    for result in graph.stream({"question": message}, config=config):
        if "prompt_prepare" in result:
            status = result["prompt_prepare"]
        elif "handle_offtopic" in result:
            status = result["handle_offtopic"]

    # Show thinking state before streaming tokens
    rows[-1] = (message, "Thinking...")
    display = _display_for_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display), gr.update(value="")

    partial = ""
    for chunk in llm.stream(status["prepared_messages"]):
        if partial == "" and chunk.content.strip() == "":
            # preliminary thinking state; already shown
            yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=thread_id), gr.update(value="")
            continue
        partial += chunk.content
        rows[-1] = (message, partial)
        display = _display_for_thread(thread_id)
        yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display), gr.update(value="")

    # Persist updated history into the graph (which the Postgres saver persists)
    try:
        current_state = graph.get_state(config)
        current_history = current_state.values.get("chat_history", [])
        new_history = current_history.copy()
        new_history.append({"role": "user", "content": status.get("prepared_question", message)})
        new_history.append({"role": "assistant", "content": partial})
        graph.update_state(
            config,
            {"chat_history": new_history, "answer": partial},
            as_node="prompt_prepare",
        )
    except Exception as e:
        print(f"Error persisting chat history: {e}")

    display = _display_for_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=list_chat_threads(), value=display), gr.update(value="")

# Gradio UI Building

with gr.Blocks(title="PolyU EEE Virtual Academic Advisor Chatbot") as demo:
    gr.Markdown("## Welcome to the PolyU EEE Virtual Academic Advising Platform!")
    gr.Text("""
    To create a new chat, enter a title and click Create. \n
    To rename an existing chat, select it from the dropdown, enter a new title, and click Rename. \n
    To delete a chat, select it from the dropdown and click Delete.
    """, label="Instructions", container=False)
    thread_state = gr.State(value={"thread_id": "", "rows": []})
    
    def init():
        tid = get_most_recent_thread_id()
        if tid:
            return load_chat(_display_for_thread(tid))
        choices = list_chat_threads()
        if choices:
            return load_chat(choices[0])
        return [], {"thread_id": "", "rows": []}, gr.update(choices=choices)
    with gr.Row():
        chat_dropdown = gr.Dropdown(choices=list_chat_threads(), label="List of Chats", allow_custom_value=True)
        with gr.Column():
            chat_name_txt = gr.Textbox(placeholder="Chat Title (optional)", label="Set / Rename Chat", container=True)
        with gr.Column():
            new_btn = gr.Button("Create", variant="primary")
            with gr.Row():
                rename_btn = gr.Button("Rename", variant="secondary")
                delete_btn = gr.Button("Delete", variant="stop")
            gr.Markdown("Note: Created chats will be saved only if you begin a conversation.")
        
    chatbot = gr.Chatbot(label="Advisor Chatbot")
    with gr.Row():
        txt = gr.Textbox(placeholder="Ask a question...", show_label=False, container=False, scale=3)
        send_btn = gr.Button("Send", scale=1, variant="huggingface")

    demo.load(init, outputs=[chatbot, thread_state, chat_dropdown])
    
    # Load on dropdown change, new chat with optional name, rename existing
    chat_dropdown.change(load_chat, inputs=[chat_dropdown], outputs=[chatbot, thread_state, chat_dropdown])
    new_btn.click(new_chat, inputs=[chat_name_txt], outputs=[chatbot, thread_state, chat_dropdown])
    rename_btn.click(rename_chat, inputs=[chat_dropdown, chat_name_txt], outputs=[chatbot, thread_state, chat_dropdown])
    delete_btn.click(delete_chat, inputs=[chat_dropdown], outputs=[chatbot, thread_state, chat_dropdown])
    txt.submit(stream_response, inputs=[txt, thread_state], outputs=[chatbot, thread_state, chat_dropdown, txt])
    send_btn.click(stream_response, inputs=[txt, thread_state], outputs=[chatbot, thread_state, chat_dropdown, txt])

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)
