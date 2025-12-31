# Dieser Code ist geschrieben Anlehung an
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
import asyncio
from typing import TypedDict, Annotated, List
import streamlit as st
import os
from langchain_core.messages import SystemMessage, AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END
from search_tool import SearchTool
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from operator import add
from langchain_core.messages import AIMessage, AIMessageChunk

load_dotenv()
FAQ = True
SYSTEM_PROMPT=("Du bist ein freundlicher Lern-Assistent. Wenn du das"
               "Such-Tool verwendest, formatiere die Quellenangaben aus den Metadaten (Feld"
               "*metadatas* im zurückgelieferten Objekt des SearchTools"
               "mit nummerierten Referenzen (z.B. [1]) im Text und der entsprechenden Quellenangabe"
               "am Ende (z.B. [1] Kapitel 3. Kuchenfiltration, S. 23)")

# Initialisiere Nachrichten
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialisiere das Basis-LLM
if "base_llm" not in st.session_state:
    st.session_state.base_llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5-mini",
        temperature=0.0,
        streaming=True
    )

# Initialisiere das Suchtool, falls im FAQ-Modus
if FAQ:
    if "tools_node" not in st.session_state:
        client = chromadb.PersistentClient(path="./chroma_db")
        emb = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="jinaai/jina-embeddings-v2-base-de")

        collection = client.get_or_create_collection(
            "verfahrenstechnik",
            embedding_function=emb)

        search_tool = SearchTool(collection)
        TOOLS = [search_tool]
        st.session_state.tools_node = ToolNode(TOOLS)
        st.session_state.llm = st.session_state.base_llm.bind_tools(TOOLS)
# Andernfalls ist das LLM das Base-LLM ohne Tools
else:
    st.session_state.llm = st.session_state.base_llm


# Track Nachrichten (messages) und speichere das LLM-Objekt, damit es
# beim Nachrichtenstreaming nicht verloren geht
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    llm: object


# Lese die Nachrichten und das LLM-Objekt aus dem Status des Graphs
# Nimm die nächste KI-Nachricht
def chat_node(state: GraphState) -> dict:
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    llm = state.get("llm")
    ai = llm.invoke(msgs)
    return {"messages": [ai]}


if "app_graph" not in st.session_state:
    graph = StateGraph(GraphState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    if FAQ:
        graph.add_node("tools", st.session_state.tools_node)
        graph.add_conditional_edges("chat", tools_condition, {"tools": "tools", "__end__": END})
        graph.add_edge("tools", "chat")
    else:
        graph.add_edge("chat", END)
    app_graph = graph.compile()
    st.session_state.app_graph = app_graph


st.title("Lern-Bot")
# Zeige, die Chat-Historie an, falls es eine gibt.
for role, content in st.session_state.messages:
    r = role if role in ("user", "assistant") else "assistant"
    with st.chat_message(r):
        st.write(content)

# RAG-Chat auf Basis von Nutzereingaben
if prompt := st.chat_input("Frag, für mehr Informationen!"):
    st.session_state.messages.append(("user", prompt))
    content = st.session_state.messages[-1][1]
    with st.chat_message("user"):
        st.write(content)

    history_msgs: List[AnyMessage] = []
    for role, content in st.session_state.messages:
        history_msgs.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))

    # Nachrichten streamen
    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty()

        for event in st.session_state.app_graph.stream({"messages": history_msgs, "llm": st.session_state.llm}, stream_mode="messages"):
            # Extract content from the event
            if isinstance(event[0], AIMessageChunk):
                chunk_content = event[0].content
                if chunk_content:
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌")

        # Finale KI-Nachricht anzeigen
        message_placeholder.markdown(full_response)

    # Finale KI-Nachricht in der Historie speichern
    st.session_state.messages.append(("assistant", full_response))
    st.rerun()

