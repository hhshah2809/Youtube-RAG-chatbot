
# 🎥 YouTube RAG Chatbot (Streamlit + LangChain + FAISS)

A **Retrieval-Augmented Generation (RAG) chatbot** that lets you **chat with any YouTube video** using its transcript.
The chatbot retrieves relevant transcript chunks using **FAISS vector search** and answers questions using an **LLM**, with **chat history support (GPT-style UI)**.

---

## 🚀 Features

* 🔗 Paste any **YouTube video URL**
* 📝 Automatically fetches **video transcript**
* 📚 Builds **FAISS vector store** from transcript
* 🔍 Semantic search using **HuggingFace embeddings**
* 🤖 Answers using **LLM (Gemini / HuggingFace models)**
* 💬 **GPT-style chat interface**
* 🧠 Maintains **chat history**
* ➕ Dynamically adds **user queries to FAISS**
* 💯 Fully local + free-tier friendly

---

## 🧠 Architecture (High Level)

```
YouTube URL
   ↓
Transcript Extraction
   ↓
Text Chunking
   ↓
Embeddings (MiniLM)
   ↓
FAISS Vector Store
   ↓
Retriever
   ↓
Prompt + Chat History
   ↓
LLM
   ↓
Answer
```

---

## 🛠️ Tech Stack

| Component  | Tool                                   |
| ---------- | -------------------------------------- |
| UI         | Streamlit                              |
| LLM        | HuggingFace / Gemini                   |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB  | FAISS                                  |
| RAG        | LangChain                              |
| Transcript | youtube-transcript-api                 |

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/youtube-rag-chatbot.git
cd youtube-rag-chatbot
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file (only required if using Gemini):

```env
GOOGLE_API_KEY=your_google_api_key
```

> 💡 Not required if you use HuggingFace models.

---

## ▶️ Run the App

```bash
streamlit run youtuberag.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works (Step-by-Step)

1. User enters **YouTube URL**
2. Transcript is fetched via `youtube-transcript-api`
3. Transcript is split into overlapping chunks
4. Chunks are embedded using **MiniLM**
5. FAISS stores embeddings
6. User asks a question
7. Relevant chunks are retrieved
8. Prompt + chat history are sent to LLM
9. LLM generates answer
10. Conversation history is updated

---

## 💬 Chat History Handling

* Stored in `st.session_state["messages"]`
* Last **2 messages** injected into prompt
* Enables **follow-up questions**
* GPT-like conversational flow

---

## 📁 Project Structure

```
.
├── youtuberag.py
├── requirements.txt
├── README.md
├── .env
└── venv/
```

---

## ⚠️ Limitations

* Depends on transcript availability
* Free LLM APIs have rate limits
* Long videos may take time to embed

---

## 🔮 Future Improvements

* 🎯 Source citations per answer
* 🗂️ Multiple video support
* 💾 Persistent FAISS storage
* 🧑‍💼 User authentication
* 🌐 Deployment on cloud

---

## 🧠 Learning Outcomes

* RAG fundamentals
* FAISS vector search
* LLM prompt engineering
* Chat history management
* Streamlit state handling

---

## 👨‍💻 Author

**Het Shah**
Built as a hands-on project to deeply understand **RAG + LLMs + Vector Databases**

---

## ⭐ If this helped you

Give the repo a ⭐ and feel free to fork & extend!

