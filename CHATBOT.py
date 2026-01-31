"""
chatbot.py
FAQ Chatbot (RAG) using LangChain + Chroma + OpenAI

Student notes:
- This script scrapes an official FAQ/help page and builds a small local vector DB.
- Then it runs a command-line chatbot that answers questions using retrieved context.

Install:
  pip install -U langchain langchain-community langchain-openai chromadb requests beautifulsoup4 tiktoken

Run:
  export OPENAI_API_KEY="YOUR_KEY"
  python chatbot.py
"""

import os
import sys
import time
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# -----------------------------
# Configuration
# -----------------------------
FAQ_FILE = "echo_faq.txt"

def build_dataset(_):
    docs = []

    with open(FAQ_FILE, "r") as f:
        text = f.read()

    parts = text.split("\n\n")

    for p in parts:
        if len(p.strip()) > 20:
            docs.append(
                Document(
                    page_content=p.strip(),
                    metadata={"source": "local_faq"}
                )
            )

    feature_schema = [
        ("page_content", "FAQ text", "string"),
        ("source", "Source of FAQ", "string"),
    ]

    stats = DatasetStats(
        num_records=len(docs),
        num_features=2,
        feature_schema=feature_schema,
    )

    return docs, stats



def build_vectorstore(docs: List[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    vs.persist()
    return vs


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def make_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful FAQ assistant for Amazon Echo/Alexa devices. "
             "Answer using ONLY the provided context. "
             "If the answer is not in the context, say you don't know and suggest where to check."),
            ("human",
             "Question: {question}\n\n"
             "Context:\n{context}\n\n"
             "Answer in 4-8 sentences, clear and simple.")
        ]
    )

    def format_docs(docs: List[Document]) -> str:
        # include sources for transparency (nice for screenshots + report)
        parts = []
        for d in docs:
            parts.append(f"- {d.page_content}\n  (source: {d.metadata.get('source')})")
        return "\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain


def ensure_index_exists() -> DatasetStats:
    # If directory exists and seems populated, skip rebuild
    if os.path.isdir(PERSIST_DIR) and any(os.scandir(PERSIST_DIR)):
        # No rebuild; dataset stats unknown here unless you stored them.
        # We'll quickly re-scrape just to print stats for the report (fast enough).
        docs, stats = build_dataset(FAQ_URLS)
        return stats

    docs, stats = build_dataset(FAQ_URLS)
    print("\n--- Dataset Stats (save this for your report) ---")
    print(f"Records (snippets): {stats.num_records}")
    print(f"Features: {stats.num_features}")
    for name, desc, dtype in stats.feature_schema:
        print(f" - {name}: {desc} ({dtype})")
    print("------------------------------------------------\n")

    print("Building vector store (first run only)...")
    build_vectorstore(docs)
    print("Vector store saved.\n")
    return stats


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Set it first, then run again.")
        sys.exit(1)

    stats = ensure_index_exists()
    vectorstore = load_vectorstore()
    rag = make_rag_chain(vectorstore)

    print("FAQ Chatbot is ready.")
    print("Type your question. Type 'exit' to quit.\n")

    # Suggest a few questions for screenshots
    print("Try these for screenshots:")
    print("  1) How does Alexa use my voice recordings?")
    print("  2) My Echo can't connect to Wi-Fi during setup. What should I do?")
    print("  3) What is Alexa Voice ID?\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        start = time.time()
        try:
            resp = rag.invoke(q)
            elapsed = time.time() - start
            print(f"\nBot ({elapsed:.2f}s): {resp.content}\n")
        except Exception as e:
            print(f"\nBot error: {e}\n")
            print("Tip: double-check your OPENAI_API_KEY and installed packages.\n")


if __name__ == "__main__":
    main()
