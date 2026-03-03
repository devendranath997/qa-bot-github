"""
QA Bot — Ask Questions About Your PDF
======================================
A Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF
and ask natural-language questions about its content.

Stack:
  - IBM WatsonX (Granite-3 LLM + Slate embeddings)
  - LangChain  (document loading, splitting, retrieval chain)
  - ChromaDB   (in-memory vector store)
  - Gradio     (web UI)

Usage:
    1. Copy .env.example to .env and fill in your WatsonX credentials.
    2. Install dependencies:  pip install -r requirements.txt
    3. Run:                   python qa_bot.py
    4. Open http://localhost:7860 in your browser.
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import gradio as gr

# ---------------------------------------------------------------------------
# Load environment variables from .env (ignored by git — see .gitignore)
# ---------------------------------------------------------------------------
load_dotenv()

WATSONX_APIKEY     = os.getenv("WATSONX_APIKEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# Validate that required credentials are present at startup
_missing = [k for k, v in {
    "WATSONX_APIKEY": WATSONX_APIKEY,
    "WATSONX_PROJECT_ID": WATSONX_PROJECT_ID,
}.items() if not v]

if _missing:
    raise EnvironmentError(
        f"Missing required environment variable(s): {', '.join(_missing)}\n"
        "Copy .env.example to .env and fill in your WatsonX credentials."
    )


# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
def get_llm() -> WatsonxLLM:
    """Return a WatsonxLLM instance using Granite as the backbone model."""
    return WatsonxLLM(
        model_id="ibm/granite-3-8b-instruct",
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "min_new_tokens": 1,
            "temperature": 0,
        },
    )


# ---------------------------------------------------------------------------
# 2. Document loader
# ---------------------------------------------------------------------------
def document_loader(file_path: str):
    """Load a PDF file and return a list of LangChain Document objects.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects, one per PDF page.
    """
    loader = PyPDFLoader(file_path)
    return loader.load()


# ---------------------------------------------------------------------------
# 3. Text splitter
# ---------------------------------------------------------------------------
def text_splitter(data):
    """Split documents into smaller, overlapping chunks for better retrieval.

    Args:
        data: List of Document objects returned by document_loader.

    Returns:
        List of smaller Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(data)


# ---------------------------------------------------------------------------
# 4. Embeddings
# ---------------------------------------------------------------------------
def watsonx_embedding() -> WatsonxEmbeddings:
    """Return a WatsonxEmbeddings instance for generating text embeddings."""
    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        url=WATSONX_URL,
        apikey=WATSONX_APIKEY,
        project_id=WATSONX_PROJECT_ID,
    )


# ---------------------------------------------------------------------------
# 5. Vector database
# ---------------------------------------------------------------------------
def vector_database(splits) -> Chroma:
    """Embed text chunks and store them in an in-memory Chroma vector store.

    Args:
        splits: List of Document chunks from text_splitter.

    Returns:
        Chroma vector store populated with embedded chunks.
    """
    embedding = watsonx_embedding()
    return Chroma.from_documents(
        documents=splits,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# 6. Retriever
# ---------------------------------------------------------------------------
def build_retriever(file_path: str):
    """Load, split, embed, and return a similarity-search retriever.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A LangChain retriever backed by a Chroma vector store.
    """
    splits = text_splitter(document_loader(file_path))
    vectordb = vector_database(splits)
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


# ---------------------------------------------------------------------------
# 7. QA chain
# ---------------------------------------------------------------------------
def retriever_qa(file, query: str) -> str:
    """Answer a query about the uploaded PDF using RAG.

    Args:
        file:  Path to the uploaded PDF file (provided by Gradio).
        query: The user's question as a plain string.

    Returns:
        The LLM's answer as a string, or an error message.
    """
    if file is None:
        return "Please upload a PDF file first."
    if not query or not query.strip():
        return "Please enter a question."

    try:
        llm = get_llm()
        retriever_obj = build_retriever(file)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False,
        )

        response = qa_chain.invoke({"query": query})
        return response["result"]

    except Exception as exc:  # pragma: no cover
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Interface:
    """Build and return the Gradio interface."""
    return gr.Interface(
        fn=retriever_qa,
        inputs=[
            gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            ),
            gr.Textbox(
                label="Your Question",
                placeholder="e.g. What is this paper about?",
                lines=2,
            ),
        ],
        outputs=gr.Textbox(label="Answer", lines=6),
        title="📄 QA Bot — Ask Questions About Your PDF",
        description=(
            "Upload any PDF document and ask natural-language questions "
            "about its content.\n\n"
            "Powered by **IBM WatsonX** (Granite-3 LLM) + "
            "**LangChain RAG** + **ChromaDB**."
        ),
        examples=[
            [None, "What is the main topic of this document?"],
            [None, "Summarize the key findings."],
            [None, "What methods were used in this study?"],
        ],
        allow_flagging="never",
    )


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",   # bind to all interfaces (needed for Docker/cloud)
        server_port=7860,
        share=False,             # set True to get a public Gradio link
    )
