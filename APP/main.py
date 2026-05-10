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
from pathlib import Path

from rag_pipeline.config import llm
from rag_pipeline.pipeline import build_graph
from rag_pipeline.memory import create_new_thread, get_chat_history, clear_thread_memory

import gr_func as cb

base_path = Path(__file__).parent
print(f"Base path: {base_path}")

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
    print(f"[GRAPH] Completed streaming response.")

    # 6. Persist updated history into the graph (which the Postgres Saver persists)
    try:
        current_state = graph.get_state(config)
        current_history = current_state.values.get("chat_history", [])
        new_history = current_history.copy()
        new_history.append({"role": "user", "content": message}) # Add user message to history.
        new_history.append({"role": "assistant", "content": partial})
        graph.update_state(
            config,
            {"chat_history": new_history, "answer": partial},
            as_node="prompt_prepare",
        )
    except Exception as e:
        print(f"Error persisting chat history: {e}")
    print(f"[GRAPH] Completed persisting chat history.")

    display = cb._display_chat_thread(thread_id)
    yield rows, {"thread_id": thread_id, "rows": rows}, gr.update(choices=cb.list_chat_threads(), value=display), gr.update(value="")

# Gradio UI Building

css_chat = """
.icon-button-wrapper.top-panel {
    display: none !important;
}
.image-container {
    padding: 20px !important;
}
.centered-markdown {
    text-align: center !important;
}
"""

with gr.Blocks(title="PolyU EEE Virtual Academic Advisor Chatbot", css=css_chat) as demo:
    gr.Markdown("# Welcome to the PolyU EEE Virtual Academic Advising Platform!", elem_classes="centered-markdown")
    
    gr.Markdown(\
    """
    **To start a new chat session, enter a chat title and click Create**. \n
    To rename an existing chat, select it from the chat list, enter a new title, and click **Rename**. \n
    To delete a chat, select it from the chat list and click **Delete**.\n
    Please chat with the advisor in **English** to get the best experience.
    """, 
    container=True)
    
    thread_state = gr.State(value={"thread_id": "", "rows": []})
    
    def init():
        """
        On app load, try to find the most recent chat thread and load it; if none found, just populate the dropdown choices.
        """
        thread_id = cb.get_most_recent_thread_id()
        if thread_id:
            display = cb._display_chat_thread(thread_id)
            return cb.load_chat(display)
        choices = cb.list_chat_threads()
        if choices:
            return cb.load_chat(choices[0])
        return [], {"thread_id": "", "rows": []}, gr.update(choices=choices, value="")
    
    # Top row UI
    with gr.Row():
        chat_dropdown = gr.Dropdown(choices=cb.list_chat_threads(), label="List of Chats", allow_custom_value=True)
        with gr.Column():
            chat_name_txt = gr.Textbox(placeholder="Chat Title", label="Set / Rename Chat", container=True)
        with gr.Column():
            new_btn = gr.Button("Create", variant="primary")
            with gr.Row():
                rename_btn = gr.Button("Rename", variant="secondary")
                delete_btn = gr.Button("Delete", variant="stop")
            # gr.Markdown("Note: Created chats will be saved **only if you begin a conversation**.")
    
    # Chatbot display and input
    chatbot = gr.Chatbot(label="Advisor Chatbot")
    with gr.Row():
        txt = gr.Textbox(placeholder="Ask a question...", show_label=False, container=False, scale=3)
        send_btn = gr.Button("Send", scale=1, variant="huggingface")
        
    with gr.Row():
        gr.Markdown("AI-generated answer may be inaccurate. **For reference only**.\nDo not share personal or sensitive information in the chat.")

    gr.HTML("""<div style="text-align: center;"><hr></div>""")
    gr.Markdown("## Introduction", elem_classes="centered-markdown")
    gr.Markdown(\
    """
    A virtual academic advisor is a chatbot implemented using an Retrieval-Augmented Generation (RAG) framework to 
    embed external documents into a database and retrieve documents based on the similarity between the input query and 
    all embedded data. An language model then uses the retrieved information to reason and generate the answer for the inquiry. 
    """, container=False, elem_classes="centered-markdown")
    
    gr.HTML("""<div style="text-align: center;"><hr></div>""")
    gr.Markdown("## Why Virtual Academic Advising?", elem_classes="centered-markdown")
    with gr.Row(variant="panel", equal_height=True):
        with gr.Column(elem_classes="image-container"):
            gr.Markdown("### Enhanced Experience", elem_classes="centered-markdown")
            gr.Markdown(
            """
            <p style='text-align: center;'>
            The chatbot provides flexible, real-time support for students in academic advising. <br>
            Traditional academic advisors’ scheme usually has limited office hours for face-to-face consultation, 
            which may not always suit students’ needs. <br>
            In a virtual academic advising chatbot that runs 24/7, students can freely choose when to consult the advisor. 
            Students can quickly receive feedback from the chatbot to resolve any concerns. 
            </p>
            """, container=True
            )
        with gr.Column(elem_classes="image-container"):
            gr.Markdown("### Enhanced Efficiency", elem_classes="centered-markdown")
            gr.Markdown(
            """
            <p style='text-align: center;'>
            The chatbot reduces the need for professors to perform basic academic advising tasks. <br>
            Traditionally, the university assigns each advisor to many students across various programmes, 
            which leads to a heavier workload and slower response times when students simultaneously seek guidance. <br>
            A computer-based virtual academic advisor can handle multiple queries at once, significantly enhancing the advising process. 
            </p>
            """, container=True
            )
        with gr.Column(elem_classes="image-container"):
            gr.Markdown("### Stay Up-to-Date", elem_classes="centered-markdown")
            gr.Markdown(
            """
            <p style='text-align: center;'>
            The chatbot can be continuously and instantly updated with new information, such as programme structures, 
            course selections, and the university’s policies, which are frequently updated. <br>
            In a virtual academic advisor, new information is easily integrated into the RAG system for embedding, 
            requiring little human intervention, significantly reducing the cost of “learning” updated information. 
            </p>
            """, container=True
            )

    gr.HTML("""<div style="text-align: center;"><hr></div>""")
    img_height = 150
    img_width = 200
    gr.Markdown("## Technical Stacks", elem_classes="centered-markdown")
    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="image-container"):
            gr.Image(f"{base_path}/assets/deepseek.png", label="DeepSeek-R1", show_label=False, height=img_height, min_width=img_width, scale=0)
            gr.Markdown("<p style='text-align: center;'><b>Large Language Model: </b>DeepSeek-R1</p>")
        with gr.Column(elem_classes="image-container"):
            gr.Image(f"{base_path}/assets/chroma.png", label="ChromaDB", show_label=False, height=img_height, min_width=img_width, scale=0)
            gr.Markdown("<p style='text-align: center;'><b>RAG Vector Database: </b>Chroma</p>")
        with gr.Column(elem_classes="image-container"):
            gr.Image(f"{base_path}/assets/bge.png", label="BGE-m3", show_label=False, height=img_height, min_width=img_width, scale=0)
            gr.Markdown("<p style='text-align: center;'><b>Text Embedding Model: </b>BGE-m3</p>")
    with gr.Row(equal_height=True):
        with gr.Column(elem_classes="image-container"):
            gr.Image(f"{base_path}/assets/langchain.png", label="LangChain", show_label=False, height=img_height, min_width=img_width, scale=0)
            gr.Markdown("<p style='text-align: center;'><b>Framework and Pipeline: </b>LangChain</p>")
        with gr.Column(elem_classes="image-container"):
            gr.Image(f"{base_path}/assets/postgresql.png", label="PostgreSQL", show_label=False, height=img_height, min_width=img_width, scale=0)
            gr.Markdown("<p style='text-align: center;'><b>Session Database: </b>PostgreSQL</p>")
        with gr.Column(elem_classes="image-container"):
            gr.Image(f"{base_path}/assets/gradio.png", label="Gradio", show_label=False, height=img_height, min_width=img_width, scale=0)
            gr.Markdown("<p style='text-align: center;'><b>GUI Framework: </b>Gradio 5</p>")

    gr.HTML("""<div style="text-align: center;"><hr></div>""")
    gr.Markdown("## Project Information", elem_classes="centered-markdown")
    gr.HTML(
        """
        <div style="text-align: center;">
            <b>Final Year Project</b>: Large Language Models with Retrieval-Augmented Generation for Virtual Academic Advising<br><br>
            <b>Student</b>: KWAN Tsz Hei Marcus (22012026D)<br>
            <b>Supervisor</b>: Prof. Man Wai MAK<br><br>
            © 2026 KWAN Tsz Hei Marcus - The Hong Kong Polytechnic University
        </div>
        """
    )

    demo.load(init, outputs=[chatbot, thread_state, chat_dropdown])
    
    # Load on dropdown change, new chat with optional name, rename existing
    chat_dropdown.change(cb.load_chat, inputs=[chat_dropdown], outputs=[chatbot, thread_state, chat_dropdown])
    new_btn.click(cb.new_chat, inputs=[chat_name_txt], outputs=[chatbot, thread_state, chat_dropdown])
    rename_btn.click(cb.rename_chat, inputs=[chat_dropdown, chat_name_txt], outputs=[chatbot, thread_state, chat_dropdown])
    delete_btn.click(cb.delete_chat, inputs=[chat_dropdown], outputs=[chatbot, thread_state, chat_dropdown])
    txt.submit(stream_response, inputs=[txt, thread_state], outputs=[chatbot, thread_state, chat_dropdown, txt])
    send_btn.click(stream_response, inputs=[txt, thread_state], outputs=[chatbot, thread_state, chat_dropdown, txt])

demo.queue(default_concurrency_limit=2)
demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, show_api=False)