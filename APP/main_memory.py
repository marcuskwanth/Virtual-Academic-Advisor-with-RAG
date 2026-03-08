"""
Gradio RAG chatbot with in-memory (LangGraph MemorySaver) persistence.
"""

import gradio as gr
from langgraph.checkpoint.memory import MemorySaver

from rag_pipeline.config import llm
from rag_pipeline.pipeline import build_graph
from rag_pipeline.memory import create_new_thread, get_chat_history, clear_thread_memory

# Build graph with in-memory checkpointer
memory = MemorySaver()
graph = build_graph(checkpointer=memory)


# Streaming response handler
def stream_response(message: str, history, thread_id: str):
    """
    Gradio streaming generator for the ChatInterface.
    Yields tuples of (partial_response, thread_id) so the thread state is propagated back through Gradio's additional_outputs mechanism.
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


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    thread_state = gr.State()

    demo = gr.ChatInterface(
        stream_response,
        additional_outputs=[thread_state],
        additional_inputs=[thread_state],
        textbox=gr.Textbox(
            placeholder="Send to the LLM...",
            container=False,
            autoscroll=True,
            scale=7,
        ),
    )
    demo.launch(debug=True)
