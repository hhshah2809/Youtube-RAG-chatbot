import streamlit as st
import os
import re
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")

st.title("🎥 YouTube RAG Chatbot")

# ---------------------------
# Extract Video ID Function
# ---------------------------
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
    return url  # assume user gave direct video ID


# ---------------------------
# Get Transcript
# ---------------------------
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



# ---------------------------
# Build Vector Store
# ---------------------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

  #  embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ---------------------------
# RAG Pipeline
# ---------------------------
def build_rag_chain(retriever):
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Use ONLY the context below to answer.
If context is insufficient, reply with "No".

Context:
{context}

Question: {question}
    """,
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    parallel = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    chain = parallel | prompt | llm | StrOutputParser()
    return chain


# ---------------------------
# Sidebar Input
# ---------------------------
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
    vector_store = build_vector_store(transcript)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    rag_chain = build_rag_chain(retriever)

    st.success("Ready! Ask anything about the video 📌")

    # Chat interface
    user_query = st.text_input("Ask a question about the video:", "")
    if user_query:
        with st.spinner("⏳ Generating answer…"):
            answer = rag_chain.invoke(user_query)
        st.markdown(f"**Answer:** {answer}")
 