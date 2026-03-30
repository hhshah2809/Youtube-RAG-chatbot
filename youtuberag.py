import streamlit as st
import os
import re
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")

st.title("🎥 YouTube RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None


# ------------------------------------
# Extract Video ID
# ------------------------------------
def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/embed/([^?]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return url


# ------------------------------------
# Get transcript
# ------------------------------------
def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        preprocessed_transcript = ""
        for item in fetched_transcript:
            preprocessed_transcript += item.text + " "
        return preprocessed_transcript.strip()
    except TranscriptsDisabled:
        return None


# ------------------------------------
# Build Vector Store
# ------------------------------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ------------------------------------
# RAG chain
# ------------------------------------
def build_rag_chain(retriever):
    prompt = PromptTemplate(
        template="""
Use ONLY the context below to answer.
If context is insufficient, reply "No".

Context:
{context}

Chat History (last 2 messages):
{history}

Question: {question}
        """,
        input_variables=["context", "history", "question"],
    )

    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            task="conversational",
            max_new_tokens=300,
            temperature=0.4,
        )
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    def last_messages():
        if "messages" not in st.session_state:
            return ""
        msgs = st.session_state["messages"]
        if len(msgs) < 2:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in msgs[-2:]])

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
            "history": RunnableLambda(lambda _: last_messages()),
        }
    )

    chain = parallel | prompt | llm | StrOutputParser()
    return chain


# ------------------------------------
# Sidebar input
# ------------------------------------
url = st.text_input("Enter YouTube Video URL:", "")

if url:
    video_id = extract_video_id(url)
    st.success(f"Extracted Video ID: {video_id}")

    st.write("⏳ Fetching transcript…")
    transcript = get_transcript(video_id)

    if transcript is None:
        st.error("Transcript disabled for this video ❌")
        st.stop()

    st.write("⏳ Building vector store…")
    st.session_state["vector_store"] = build_vector_store(transcript)
    retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 3})
    rag_chain = build_rag_chain(retriever)

    st.success("Ready! Ask anything about the video 📌")

    # ---------------------------
    # Chat UI + History
    # ---------------------------
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['content']}")

    user_query = st.chat_input("Ask a question…")

    if user_query:
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # store only user query in FAISS
        st.session_state["vector_store"].add_texts(
            [user_query],
            embeddings=[embeddings.embed_query(user_query)]
        )

        with st.spinner("⏳ Generating answer…"):
            answer = rag_chain.invoke(user_query)

        st.session_state["messages"].append({"role": "assistant", "content": answer})

        st.rerun()
