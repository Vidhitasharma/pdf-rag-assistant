# --- Imports ---
import PyPDF2
import torch
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import gradio as gr

# Global variables for retriever & chain
retriever = None
qa_chain = None


# --- Step 1: Function to Process PDF ---
def process_pdf(pdf_file):
    global retriever, qa_chain

    if pdf_file is None:
        return "⚠️ Please upload a PDF first."

    # Read PDF
    all_text = []
    reader = PyPDF2.PdfReader(pdf_file.name)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)

    full_text = "\n".join(all_text)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    chunk_embeddings = embedding_model.embed_documents(chunks)

    # ChromaDB
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection(name="pdf_embeddings")
    except:
        pass

    collection = chroma_client.create_collection(name="pdf_embeddings")

    collection.add(
        embeddings=chunk_embeddings,
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name="pdf_embeddings",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load Groq Model
    if "GROQ_API_KEY" not in os.environ:
        return "❌ Please set your GROQ_API_KEY first. Visit: https://console.groq.com/"

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512
    )

    # Build QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    return "✅ PDF uploaded & processed successfully! You can now ask questions."


# --- Step 2: Function to Answer Questions ---
def answer_question(query):
    global qa_chain
    if qa_chain is None:
        return "⚠️ Please upload a PDF first."
    result = qa_chain({"query": query})
    return result["result"]


# --- Step 3: Build Gradio UI ---
with gr.Blocks() as ui:
    gr.Markdown("## 🤖 PDF Q&A Chatbot")
    gr.Markdown("Upload a PDF and ask questions about it.")

    pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
    process_btn = gr.Button("📥 Process PDF")
    process_status = gr.Textbox(label="Status")

    question_box = gr.Textbox(label="Ask your question:")
    answer_box = gr.Textbox(label="Answer:")
    ask_btn = gr.Button("Ask")

    process_btn.click(fn=process_pdf, inputs=pdf_input, outputs=process_status)
    ask_btn.click(fn=answer_question, inputs=question_box, outputs=answer_box)

ui.launch()
