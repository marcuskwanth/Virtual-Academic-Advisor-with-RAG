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
from rag_pipeline.memory import create_new_thread, get_chat_history, clear_thread_memory, get_chat_history_text

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

def list_threads() -> list:
    """
    Retrieve all distinct thread IDs from the checkpoints table.
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Error listing threads: {e}")
        return []


# Streaming response handler

def stream_response(message: str, history, thread_id: str):
    """
    Gradio streaming generator for the ChatInterface.
    Yields tuples of (partial_response, thread_id) to propagate the thread state back through Gradio's additional_outputs mechanism.
    """
    if not thread_id:
        thread_id = create_new_thread()

    config = {"configurable": {"thread_id": thread_id}}
    yield ("Query received. Processing...", thread_id)

    # Run the graph to obtain prepared prompt messages
    status = None
    for result in graph.stream({"question": message}, config=config):
        if "prompt_prepare" in result:
            status = result["prompt_prepare"]
        elif "handle_offtopic" in result:
            status = result["handle_offtopic"]

    # Stream the LLM response token-by-token
    partial = ""
    for chunk in llm.stream(status["prepared_messages"]):
        if partial == "" and chunk.content.strip() == "":
            yield ("Thinking...", thread_id)
            continue
        partial += chunk.content
        yield (partial, thread_id)

    # Persist chat history into the graph state
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
    yield (partial, thread_id)


# Gradio interface

if __name__ == "__main__":
    thread_state = gr.State()

    with gr.Blocks(title="Virtual Academic Advisor Chatbot") as demo:
        with gr.Tabs():
            with gr.TabItem("Chat"):
                gr.ChatInterface(
                    stream_response,
                    additional_outputs=[thread_state],
                    additional_inputs=[thread_state],
                    textbox=gr.Textbox(
                        placeholder="Send",
                        container=False,
                        autoscroll=True,
                        scale=7,
                    ),
                )
            with gr.TabItem("History"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    thread_dropdown = gr.Dropdown(
                        choices=[], label="Select Thread", interactive=True
                    )

                history_display = gr.Textbox(
                    label="Chat History", lines=20, interactive=False
                )

                with gr.Row():
                    new_thread_btn = gr.Button("Create New Chat")
                    clear_thread_btn = gr.Button("Clear Selected Chat")

                # History tab callbacks
                def refresh_threads():
                    return gr.Dropdown(choices=list_threads())

                def load_history(thread_id: str) -> str:
                    return get_chat_history_text(graph, thread_id) if thread_id else ""

                def create_new_thread_ui():
                    new_id = create_new_thread()
                    return (
                        gr.Dropdown(choices=list_threads(), value=new_id),
                        get_chat_history_text(graph, new_id),
                    )

                def clear_thread_ui(thread_id: str) -> str:
                    if thread_id:
                        clear_thread_memory(graph, thread_id)
                        return get_chat_history_text(graph, thread_id)
                    return ""

                refresh_btn.click(refresh_threads, outputs=thread_dropdown)
                thread_dropdown.change(
                    load_history, inputs=thread_dropdown, outputs=history_display
                )
                new_thread_btn.click(
                    create_new_thread_ui, outputs=[thread_dropdown, history_display]
                )
                clear_thread_btn.click(
                    clear_thread_ui, inputs=thread_dropdown, outputs=history_display
                )

    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
