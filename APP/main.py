"""
Gradio RAG chatbot with PostgreSQL (LangGraph PostgresSaver) persistence.

Prerequisites:
  1. Set DB_URI env var / ensure the default credentials match your setup.
  2. Ensure PostgreSQL server is running and accessible (2 tables: "checkpoints" and "threads").

Environment variables:
  DB_URI: PostgreSQL connection string
"""

import os
import gradio as gr
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool

from rag_pipeline.config import llm
from rag_pipeline.pipeline import build_graph
from rag_pipeline.memory import create_new_thread, get_chat_history, clear_thread_memory

import gr_func as cb

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

cb.setup(graph, pool)

# Graph node function

def stream_response(message: str, thread_state_obj: dict):
    """
    Generator that streams assistant tokens and persists history. `thread_state_obj` is a dict with keys `thread_id` and `rows`.
    This function yields four outputs: (chat_rows, thread_state_obj, dropdown_update, textbox_update).
    """
    thread_id = (thread_state_obj or {}).get("thread_id")
    rows = (thread_state_obj or {}).get("rows", []) or []

    # 1. Ensure have a thread id
    if not thread_id:
        thread_id = create_new_thread()

    config = {"configurable": {"thread_id": thread_id}}

    # 2. Add the user message and an initial assistant placeholder
    rows = rows + [(message, "Query received. Processing...")]
    display = cb._display_chat_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=cb.list_chat_threads(), value=display), gr.update(value="")

    # 3. Run graph pipeline to prepare the prompt
    status = None
    for result in graph.stream({"question": message}, config=config):
        if "prompt_prepare" in result:
            status = result["prompt_prepare"]
        elif "handle_offtopic" in result:
            status = result["handle_offtopic"]

    # 4. Show thinking state (as in DeepSeek) before streaming tokens
    print(f"[GRAPH] Graph execution completed, preparing to stream LLM response.")
    rows[-1] = (message, "Thinking...")
    display = cb._display_chat_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=cb.list_chat_threads(), value=display), gr.update(value="")

    # 5. Stream LLM response token by token, updating the last assistant message in the chat
    partial = ""
    for chunk in llm.stream(status["prepared_messages"]):
        if partial == "" and chunk.content.strip() == "":
            # preliminary thinking state; already shown
            # yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=cb.list_chat_threads(), value=thread_id), gr.update(value="")
            continue
        partial += chunk.content
        rows[-1] = (message, partial)
        display = cb._display_chat_thread(thread_id)
        yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=cb.list_chat_threads(), value=display), gr.update(value="")

    # 6. Persist updated history into the graph (which the Postgres Saver persists)
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

    display = cb._display_chat_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=cb.list_chat_threads(), value=display), gr.update(value="")

# Gradio UI Building

with gr.Blocks(title="PolyU EEE Virtual Academic Advisor Chatbot") as demo:
    gr.Markdown("## Welcome to the PolyU EEE Virtual Academic Advising Platform!")
    gr.Text(\
    """
    To create a new chat, enter a chat title and click Create. \n
    To rename an existing chat, select it from the chat list, enter a new title, and click Rename. \n
    To delete a chat, select it from the chat list and click Delete.
    """, 
    label="Instructions", container=False)
    
    thread_state = gr.State(value={"thread_id": "", "rows": []})
    
    def init():
        """
        On app load, try to find the most recent chat thread and load it; if none found, just populate the dropdown choices.
        """
        thread_id = cb.get_most_recent_thread_id()
        if thread_id:
            return cb.load_chat(cb._display_chat_thread(thread_id))
        choices = cb.list_chat_threads()
        if choices:
            return cb.load_chat(choices[0])
        return [], {"thread_id": "", "rows": []}, gr.update(choices=choices)
    
    # Top row UI
    with gr.Row():
        chat_dropdown = gr.Dropdown(choices=cb.list_chat_threads(), label="List of Chats", allow_custom_value=True)
        with gr.Column():
            chat_name_txt = gr.Textbox(placeholder="Chat Title (optional)", label="Set / Rename Chat Title", container=True)
        with gr.Column():
            new_btn = gr.Button("Create", variant="primary")
            with gr.Row():
                rename_btn = gr.Button("Rename", variant="secondary")
                delete_btn = gr.Button("Delete", variant="stop")
            gr.Markdown("Note: Created chats will be saved only if you begin a conversation.")
    
    # Chatbot display and input
    chatbot = gr.Chatbot(label="Advisor Chatbot")
    with gr.Row():
        txt = gr.Textbox(placeholder="Ask a question...", show_label=False, container=False, scale=3)
        send_btn = gr.Button("Send", scale=1, variant="huggingface")
        
    with gr.Row():
        gr.Markdown("AI-generated answer may be inaccurate. For reference only.\nDo not share personal or sensitive information in the chat.")

    demo.load(init, outputs=[chatbot, thread_state, chat_dropdown])
    
    # Load on dropdown change, new chat with optional name, rename existing
    chat_dropdown.change(cb.load_chat, inputs=[chat_dropdown], outputs=[chatbot, thread_state, chat_dropdown])
    new_btn.click(cb.new_chat, inputs=[chat_name_txt], outputs=[chatbot, thread_state, chat_dropdown])
    rename_btn.click(cb.rename_chat, inputs=[chat_dropdown, chat_name_txt], outputs=[chatbot, thread_state, chat_dropdown])
    delete_btn.click(cb.delete_chat, inputs=[chat_dropdown], outputs=[chatbot, thread_state, chat_dropdown])
    txt.submit(stream_response, inputs=[txt, thread_state], outputs=[chatbot, thread_state, chat_dropdown, txt])
    send_btn.click(stream_response, inputs=[txt, thread_state], outputs=[chatbot, thread_state, chat_dropdown, txt])

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)
