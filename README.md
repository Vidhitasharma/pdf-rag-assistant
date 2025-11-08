# pdf-rag-assistant
An interactive PDF Question-Answering chatbot built using RAG, Chroma Vector DB, Groq Llama 3.3 model, and Gradio UI. Upload any PDF and ask questions in natural language.
# 🤖 PDF Q&A Chatbot (RAG + Groq + Gradio)

A smart document assistant that allows you to **upload any PDF** and interact with it conversationally.  
You can ask questions, extract important details, summarize content, and explore documents naturally — just like chatting with your PDF.

This system uses **RAG (Retrieval-Augmented Generation)** to pull relevant text from the document and generate responses through a powerful language model.  
Everything is wrapped in an easy-to-use **Gradio** web interface that runs smoothly in **Google Colab**.

---

## 🚀 Features

- 📄 Upload *any* PDF directly through the UI  
- 🔍 Automatically extracts and organizes PDF text  
- ✂️ Smart text chunking for improved context understanding  
- 🧠 RAG-based accurate question answering  
- 💬 Clean and simple Gradio Chat UI  
- ☁️ Runs directly in Google Colab (no setup needed)  

---

## 🧰 Tech Stack

| Component        | Technology / Tool                       |
|-----------------|------------------------------------------|
| Retrieval Model | RAG (Retrieval-Augmented Generation)     |
| Embeddings      | `intfloat/multilingual-e5-base` (HF)     |
| Vector Storage  | Chroma Vector Database                   |
| PDF Processing  | PyPDF2                                   |
| User Interface  | Gradio                                   |
| Framework       | LangChain                                |

---

## 🖥 How It Works

1. Upload your PDF through the UI  
2. The text is extracted and split into meaningful chunks  
3. Chunks are embedded and stored in a vector database  
4. When you ask a question:
   - The system searches for the most relevant text chunks  
   - Then generates a clear and accurate reply  

---

## 📦 Running the Project

1. Open the notebook in **Google Colab**
2. Run all cells in order
3. Upload a PDF using the UI
4. Type your question and get your answer instantly ✅

No external setup required.  
No local installation.  
Everything runs in your browser.

---

## ✨ Example Use Cases

| Use Case | Description |
|---------|-------------|
| Resume Understanding | Ask questions about your own resume |
| Research Papers | Summarize sections or find key arguments |
| Books / Notes | Turn reading into an interactive experience |
| Company Docs | Extract policies, roles, instructions |

---

## 🙌 Author
Built with ❤️
Feel free to fork, improve, and contribute.


