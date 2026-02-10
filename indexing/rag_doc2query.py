import os
from typing import List
from haystack import Pipeline, component, Document
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.google_genai import (
    GoogleGenAIChatGenerator,
)
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack.dataclasses import ChatMessage

import local_config as config

# --- Custom Component: Document Joiner ---

@component
class DocumentJoiner:
    """
    Joins documents from the factual and synthetic retrievers.
    For now, performs basic concatenation.
    """
    @component.output_types(documents=List[Document])
    def run(self, fact_docs: List[Document], synthetic_docs: List[Document]):
        # Tagging for visibility in the demo printout
        for doc in fact_docs:
            doc.meta["origin"] = "factual"
        for doc in synthetic_docs:
            doc.meta["origin"] = "synthetic"
            # Optional: If you want the LLM to see the original factual text 
            # instead of the lazy query, you'd swap content here.
            # doc.content = doc.meta.get("original_text", doc.content)
        return {"documents": fact_docs + synthetic_docs}

# --- Pipeline Setup ---

# 1. Initialize two stores (one per namespace)
fact_store = PineconeDocumentStore(
    api_key=Secret.from_token(config.PINECONE_API_KEY),
    index=config.PINECONE_INDEX_NAME,
    namespace="philosophy-doc2query",
    dimension=384,
)

synthetic_store = PineconeDocumentStore(
    api_key=Secret.from_token(config.PINECONE_API_KEY),
    index=config.PINECONE_INDEX_NAME,
    namespace="philosophy-doc2query-synthetic",
    dimension=384,
)

# 2. Define the RAG Prompt
template = [ChatMessage.from_user("""
        Answer the question based on the provided context from the Stanford Encyclopedia of Philosophy.
        The context contains both factual excerpts and synthetic intent matches.
        
        If the answer isn't in the context, explain what you did find.

        Context:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """)]

rag_pipe = Pipeline()

# Shared Embedder
rag_pipe.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)

# Dual Retrievers
rag_pipe.add_component(
    "fact_retriever", PineconeEmbeddingRetriever(document_store=fact_store, top_k=10)
)
rag_pipe.add_component(
    "synthetic_retriever", PineconeEmbeddingRetriever(document_store=synthetic_store, top_k=10)
)

# Joiner
rag_pipe.add_component("joiner", DocumentJoiner())

# Generator & Prompt
rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))
rag_pipe.add_component(
    "llm",
    GoogleGenAIChatGenerator(
        api_key=Secret.from_token(config.GEMINI_API_KEY),
        model="gemini-2.5-flash-lite",
    ),
)

# --- 4. Connect the dots ---

# Connect Embedder to both Retrievers
rag_pipe.connect("embedder.embedding", "fact_retriever.query_embedding")
rag_pipe.connect("embedder.embedding", "synthetic_retriever.query_embedding")

# Connect Retrievers to Joiner
rag_pipe.connect("fact_retriever.documents", "joiner.fact_docs")
rag_pipe.connect("synthetic_retriever.documents", "joiner.synthetic_docs")

# Connect Joiner to Prompt
rag_pipe.connect("joiner.documents", "prompt_builder.documents")
rag_pipe.connect("prompt_builder.prompt", "llm.messages")

rag_pipe.warm_up()

# --- 5. Run a test query ---

if __name__ == "__main__":
    while True:
        message = input("Enter a question: ")
        if message.strip().lower() in ("exit()", "quit()"):
            break

        query = message.strip()
        print(f"\nDual-Path Querying: {query}...")

        result = rag_pipe.run(
            data={"embedder": {"text": query}, "prompt_builder": {"question": query}},
            include_outputs_from=["joiner"],
        )

        print("\n--- RESPONSE ---")
        print(result["llm"]["replies"][0].text)
        
        print("\n--- MERGED SOURCES ---")
        for doc in result["joiner"]["documents"]:
            origin = doc.meta.get("origin", "unknown")
            chunk_id = doc.meta.get("parent_chunk_id") or doc.id
            title = doc.meta.get("title") or doc.meta.get("source_title")
            score = doc.score if doc.score else 0.0
            print(f"[{origin.upper()}] {title}-{chunk_id} [score:{score:.3f}]")
        print("\n", "------" * 10, "\n")