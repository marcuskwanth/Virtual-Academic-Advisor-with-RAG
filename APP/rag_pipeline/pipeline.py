"""
RAG LangGraph pipeline: State definition, graph node functions, and build_graph().
The same pipeline is shared by both app.py versions (in-memory + PostgreSQL).
"""

import math
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

from .config import (
    simpler_llm,
    vectorStore,
    bm25_retriever,
    colbert,
    USE_COLBERT,
    num_queries,
    num_docs,
    num_immediate_docs_rrf,
    num_immediate_docs_colbert,
    num_chat_his,
)
from .prompts import (
    query_classification_prompt,
    offtopic_prompt,
    query_tra_prompt,
    query_gen_prompt,
    prompt,
)

# Graph state

class State(TypedDict, total=False):
    question: str                    # raw user input
    query_type: str                  # "DOMAIN" or "OFFTOPIC"
    contextualized_question: str     # standalone rewritten query
    queries: List[str]               # alternative queries (RAG-Fusion)
    context: List[Document]          # retrieved documents
    prepared_messages: List[dict]    # formatted prompt ready for llm.stream()
    prepared_question: str           # question used for prompt (stored in history)
    answer: str                      # final streamed answer
    chat_history: List[dict]         # running conversation history

# Routing

def classify_query(state: State) -> dict:
    messages = query_classification_prompt.invoke({"question": state["question"]})
    response = simpler_llm.invoke(messages)
    query_type = "DOMAIN" if "DOMAIN" in response.content else "OFFTOPIC"
    print(f"[Router] Query classified as: {query_type}")
    return {"query_type": query_type}

def route_query(state: State) -> str:
    return state.get("query_type", "DOMAIN")

# Query contextualisation

def contextualize_question(state: State) -> dict:
    question = state["question"]
    chat_history = state.get("chat_history", [])

    if not chat_history:
        print(f"[Query Transform] No history – using original: {question}")
        return {"contextualized_question": question}

    history = "".join(
        f"{msg['role'].capitalize()}: {msg['content']}\n\n"
        for msg in chat_history[-num_chat_his:]
    )
    messages = query_tra_prompt.invoke({"chat_history": history, "question": question})
    response = simpler_llm.invoke(messages)
    contextualized_question = response.content.strip()

    print(f"[Query Transform] Original:       {question}")
    print(f"[Query Transform] Contextualized: {contextualized_question}")
    return {"contextualized_question": contextualized_question}

# RAG-Fusion retrieval & reranking

def swap_table_if_exists(doc: Document) -> Document:
    """
    If the document chunk has an 'original_table' in metadata, swap the page_content to that.
    """
    
    if "original_table" in doc.metadata:
        print("[Table found] Swapping in original table content for document chunk")
        new_result = doc.metadata["original_table"]
    else:
        new_result = doc.page_content
    return Document(
        page_content=new_result,
        metadata=doc.metadata,
    )

def generate_queries(state: State) -> dict:
    """
    Generate multiple alternative queries for RAG-Fusion.
    """
    question = state.get("contextualized_question", state["question"])
    messages = query_gen_prompt.invoke({"question": question, "num_queries": num_queries})
    response = simpler_llm.invoke(messages)
    queries = [q for q in response.content.strip().split("\n") if q.strip()]
    print(f"[RRF] Generated {len(queries)} queries")
    return {"queries": queries}
    
def retrieve_ragfusion(state: State) -> dict:
    """
    Retrieve candidate documents for every generated query.
    """
    all_docs = []
    print(f"[RRF] Retrieving for {len(state['queries'])} queries...\n")
    for idx, query in enumerate(state["queries"], 1):
        print(f"[RRF] Query {idx}: {query}")
        
        # BM25 retrieval
        bm25_docs = bm25_retriever.get_relevant_documents(query)[:num_immediate_docs_rrf]
        print(f"[RRF] BM25 retrieved {len(bm25_docs)} docs")
        all_docs.append(bm25_docs)          # first ranked list for RRF
        
        # Cos-sim retrieval
        retrieved_with_scores = vectorStore.similarity_search_with_score(query, k=num_immediate_docs_rrf)
        retrieved_docs = [doc for doc, _ in retrieved_with_scores]
        print(f"[RRF] Vector search retrieved {len(retrieved_docs)} docs")
        all_docs.append(retrieved_docs)     # second ranked list for RRF
        
    return {"context": all_docs}

def rrf_ragfusion(state: State, k: int = 60) -> dict:
    """
    Fuse and rerank documents using Reciprocal Rank Fusion.
    """
    fused_scores: dict = {}
    for docs in state["context"]:
        for rank, doc in enumerate(docs):
            # Use a more unique key: (content, source)
            key = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("chunk_id", ""))
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + k)

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Reconstruct document mapping using the same key
    doc_map = {
        (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("chunk_id", "")): doc
        for docs in state["context"]
        for doc in docs
    }
    reranked_docs = [doc_map[key] for key, _ in reranked[:num_docs]]
    
    # reranked_docs = [swap_table_if_exists(doc) for doc in reranked_docs]
    
    print(f"[RRF] Selected top {len(reranked_docs)} documents after fusion")
    return {"context": reranked_docs}

# ColBERT retrieval & reranking

def retrieve_colbert(state: State) -> dict:
    """
    Retrieve documents using initial vector search then ColBERT reranking.
    """
    question = state.get("contextualized_question", state["question"])
    
    # BM25 retrieval
    bm25_docs = bm25_retriever.get_relevant_documents(question)[:num_immediate_docs_colbert]
    
    # Cos-sim retrieval
    retrieval_with_scores = vectorStore.similarity_search_with_score(question, k=num_immediate_docs_colbert)
    retrieved_docs = [doc for doc, _ in retrieval_with_scores]
    
    # Merge & de-duplicate
    doc_map = {}
    for doc in bm25_docs + retrieved_docs:
        key = (doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("chunk_id", ""))
        doc_map.setdefault(key, doc)

    docs = list(doc_map.values())
    doc_texts = [doc.page_content for doc in docs]
    print(f"[ColBERT] {len(docs)} initial documents to rerank with ColBERT")
    
    reranked_results = colbert.rerank(query=question, documents=doc_texts, k=num_docs)
    
    # Check for NaN scores in reranked results, which can occur with ColBERT and cause issues
    checked_results = []
    for result in reranked_results:
        # Check if score is NaN
        if "score" in result and math.isnan(result["score"]):
            print(f"[ColBERT] Warning: Skipping result with NaN score: {result.get('content', '')[:50]}...")
            continue
        checked_results.append(result)

    # Match reranked results back to a docs list
    finalized_docs = []
    for result in checked_results:
        for doc in docs:
            if doc.page_content == result["content"]:
                finalized_docs.append(doc)
                break

    # If no valid docs after checking, fall back to initial results
    if not finalized_docs:
        print(f"[ColBERT] Warning: No valid reranked results, using top initial candidates")
        finalized_docs = docs[:num_docs]
        
    #finalized_docs = [swap_table_if_exists(doc) for doc in finalized_docs]

    print(f"[ColBERT] Reranked to top {len(finalized_docs)} documents")
    return {"context": finalized_docs}

# Prompt preparation

def prompt_prepare(state: State) -> dict:
    """
    Format the final prompt from retrieved context and chat history.
    """
    # Prepend source information to context chunks to help LLM recognize structure and source
    formatted_context = []
    for doc in state["context"]:
        content_to_append = \
        """
        ------------ Next Document ------------\n
        --- Retrieved from: {source} ---\n
        {content}\n
        ------------ End of Document ------------\n
        """.format(
            source=doc.metadata.get("source", "N/A"),
            content=doc.page_content
        )
        
        formatted_context.append(content_to_append)

    docs_content = "\n".join(formatted_context)

    if state.get("chat_history"):
        chat_history_str = "".join(
            f"{msg['role'].capitalize()}: {msg['content']}\n\n"
            for msg in state["chat_history"][-(3*2):]   # last 3 message pairs (user+assistant) for context
        )
    else:
        chat_history_str = "No previous conversation."

    messages = prompt.invoke({
        "question": state.get("contextualized_question", state["question"]),
        "context": docs_content,
        "chat_history": chat_history_str,
    })
    return {
        "prepared_messages": messages,
        "prepared_question": state.get("contextualized_question", state["question"]),
    }

# Off-topic handling

def handle_offtopic(state: State) -> dict:
    """
    Produce a polite off-topic response without retrieval.
    """
    question = state["question"]
    chat_history = state.get("chat_history", [])

    history_str = (
        "".join(
            f"{msg['role'].capitalize()}: {msg['content']}\n\n"
            for msg in chat_history[-3:]
        )
        if chat_history
        else "No previous conversation."
    )
    messages = offtopic_prompt.invoke({"question": question, "chat_history": history_str})
    return {"prepared_messages": messages, "prepared_question": question}

# Graph builder

def build_graph(checkpointer, store=None):
    """
    Build and compile the RAG LangGraph.
    """
    graph_builder = StateGraph(State)

    # Routing nodes
    graph_builder.add_node("classify_query", classify_query)
    graph_builder.add_node("handle_offtopic", handle_offtopic)

    # Shared nodes
    graph_builder.add_node("contextualize_question", contextualize_question)
    graph_builder.add_node("prompt_prepare", prompt_prepare)

    # Retrieval nodes
    if USE_COLBERT:
        graph_builder.add_node("retrieve_colbert", retrieve_colbert)
    else:
        graph_builder.add_node("generate_queries", generate_queries)
        graph_builder.add_node("retrieve", retrieve_ragfusion)
        graph_builder.add_node("fuse_and_rerank", rrf_ragfusion)

    # Edges
    graph_builder.add_edge(START, "classify_query")
    graph_builder.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "DOMAIN": "contextualize_question",
            "OFFTOPIC": "handle_offtopic",
        },
    )

    if USE_COLBERT:
        graph_builder.add_edge("contextualize_question", "retrieve_colbert")
        graph_builder.add_edge("retrieve_colbert", "prompt_prepare")
    else:
        graph_builder.add_edge("contextualize_question", "generate_queries")
        graph_builder.add_edge("generate_queries", "retrieve")
        graph_builder.add_edge("retrieve", "fuse_and_rerank")
        graph_builder.add_edge("fuse_and_rerank", "prompt_prepare")

    # Compile
    compile_kwargs = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store

    graph = graph_builder.compile(**compile_kwargs)

    print("Graph compiled. Pipeline flow:")
    print("  1. classify_query")
    print("  2a. [DOMAIN]   → contextualize_question → retrieve → prompt_prepare")
    print("  2b. [OFFTOPIC] → handle_offtopic (terminal)")

    return graph
