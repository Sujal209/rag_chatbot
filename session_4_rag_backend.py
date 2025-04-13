#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Chatbot Backend Module

Using Gemini API, LangChain, and ChromaDB for RAG.
"""

import os
import pdfplumber
import google.generativeai as genai
from typing import List, Dict, Tuple, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma


# ----------------------------------------
# Setup and Configuration
# ----------------------------------------

def setup_api_key(api_key: str) -> None:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    print("✅ API key configured")


# ----------------------------------------
# PDF Handling
# ----------------------------------------

def upload_pdf(pdf_path: str) -> Optional[str]:
    if os.path.exists(pdf_path):
        print(f"✅ PDF file found at: {pdf_path}")
        return pdf_path
    else:
        print(f"❌ File not found: {pdf_path}")
        return None


def parse_pdf(pdf_path: str) -> Optional[str]:
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        print(f"✅ PDF parsed. Extracted {len(text)} characters")
        return text
    except Exception as e:
        print(f"❌ Error parsing PDF: {e}")
        return None


# ----------------------------------------
# Text Chunking
# ----------------------------------------

def create_document_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        print(f"✅ Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"❌ Error splitting text: {e}")
        return []


# ----------------------------------------
# Embedding
# ----------------------------------------

def init_embedding_model(model_name: str = "models/text-embedding-004") -> Optional[GoogleGenerativeAIEmbeddings]:
    try:
        model = GoogleGenerativeAIEmbeddings(model=model_name)
        print("✅ Embedding model initialized")
        return model
    except Exception as e:
        print(f"❌ Error initializing embedding model: {e}")
        return None


def embed_documents(embedding_model: GoogleGenerativeAIEmbeddings, text_chunks: List[str]) -> bool:
    try:
        if not text_chunks:
            print("❌ No chunks to embed")
            return False
        test_embedding = embedding_model.embed_query(text_chunks[0][:100])
        if test_embedding:
            print("✅ Embedding test passed")
            return True
        else:
            print("❌ Embedding test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        return False


# ----------------------------------------
# Vector DB (ChromaDB)
# ----------------------------------------

def store_embeddings(
    embedding_model: GoogleGenerativeAIEmbeddings,
    text_chunks: List[str],
    collection_name: str = "default_collection",
    persist_directory: str = "./chroma_db",
    metadatas: Optional[List[Dict[str, str]]] = None
) -> Optional[Chroma]:
    try:
        print(f"🧠 Storing {len(text_chunks)} chunks to ChromaDB...")
        vectorstore = Chroma.from_texts(
            texts=text_chunks,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
            metadatas=metadatas
        )
        vectorstore.persist()
        print("✅ Vector database created and saved")
        return vectorstore
    except Exception as e:
        print(f"❌ Failed to create vector database: {e}")
        return None


# ----------------------------------------
# Context & Querying
# ----------------------------------------

def get_context_from_chunks(relevant_chunks, splitter="\n\n---\n\n") -> str:
    return splitter.join(
        f"[Chunk {i+1}]: {chunk.page_content}" for i, chunk in enumerate(relevant_chunks)
        if hasattr(chunk, "page_content")
    )


def query_with_full_context(
    query: str,
    vectorstore: Chroma,
    model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
    k: int = 3,
    temperature: float = 0.3
) -> Tuple[str, str, List[Any]]:
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        relevant_chunks = retriever.get_relevant_documents(query)
        context = get_context_from_chunks(relevant_chunks)

        prompt = f"""You are a helpful assistant using the provided context to answer questions.

Context:
{context}

Question: {query}

Answer:"""

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=1024
        )

        response = llm.invoke(prompt)
        return response.content, context, relevant_chunks

    except Exception as e:
        print(f"❌ Error during querying: {e}")
        return f"Error: {e}", "", []
