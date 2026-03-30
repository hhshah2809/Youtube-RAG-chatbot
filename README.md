# 🚀 YouTube RAG Chatbot

![GitHub stars](https://img.shields.io/github/stars/hhshah2809/Youtube-RAG-chatbot?style=social)
![GitHub forks](https://img.shields.io/github/forks/hhshah2809/Youtube-RAG-chatbot?style=social)
![License](https://img.shields.io/badge/license-MIT-green)
![Tech](https://img.shields.io/badge/stack-MERN%20%2B%20AI-blue)

---

## 📌 Overview

An intelligent chatbot that answers user queries based on **YouTube video content** using **Retrieval-Augmented Generation (RAG)**.

It extracts transcripts, converts them into embeddings, and retrieves the most relevant context to generate accurate responses using LLMs.

---

## ✨ Features

* 🎥 Extract YouTube transcripts
* 🔍 Semantic search using embeddings
* 🤖 Context-aware AI responses
* ⚡ Fast retrieval with vector database
* 🔐 Secure API key handling
* 📦 Modular full-stack architecture

---

## 🧠 Tech Stack

### 🤖 AI / RAG Pipeline

* Python
* Hugging Face Transformers
* FAISS (Vector DB)
* Streamlit
---

## 📂 Project Structure

```
RAG/
├── frontend/        # React frontend
├── backend/         # Node.js backend
├── rag/             # Python RAG pipeline
├── .env.example     # Environment variables template
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/hhshah2809/Youtube-RAG-chatbot.git
cd Youtube-RAG-chatbot
```

---

### 2️⃣ Install Dependencies


#### Python (RAG)

```bash
cd ../rag
pip install -r requirements.txt
```

---

### 3️⃣ Environment Variables

Create a `.env` file in root:

```env
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
MONGO_URI=your_mongodb_uri
PORT=5000
```

---

### 4️⃣ Run the Project

#### ▶️ RAG Pipeline

```bash
cd rag
streamlit app.py
```

---

## 🔐 Security

* ❌ Never push `.env` files
* ✅ Use `.env.example`
* 🔁 Rotate API keys regularly

---

## 📸 Demo

*Add screenshots or demo video here*

---

## 🚀 Future Enhancements

* 🌐 Deployment (AWS / Vercel / Docker)
* 🧾 Multi-video querying
* 🎙️ Voice-based queries
* 📊 Analytics dashboard

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first.

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Het Shah**

---

## ⭐ Support

If you like this project:
👉 Star the repo
👉 Share it with others

---
