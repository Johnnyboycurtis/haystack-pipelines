import os
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.google_genai import (
    GoogleGenAIChatGenerator,
)
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder

import local_config as config

# 1. Initialize the Document Store pointing to the FACTUAL namespace
document_store = PineconeDocumentStore(
    api_key=Secret.from_token(config.PINECONE_API_KEY),
    index=config.PINECONE_INDEX_NAME,
    namespace="philosophy-doc2query",  # Ignoring the synthetic namespace
    dimension=384,
)

# 2. Define the RAG Prompt
# 2. Define the RAG Prompt as a list of ChatMessages
template = [ChatMessage.from_user("""
        Answer the question based on the provided context from the Stanford Encyclopedia of Philosophy.
        If the answer isn't in the context, explain what you did find (e.g. I didn't find what you asked for but I did find...)

        Context:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        Question: {{ question }}
        Answer:
        """)]

# 3. Build the Pipeline
rag_pipe = Pipeline()

rag_pipe.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
rag_pipe.add_component(
    "retriever", PineconeEmbeddingRetriever(document_store=document_store, top_k=10)
)

# Use ChatPromptBuilder instead of PromptBuilder
rag_pipe.add_component("prompt_builder", ChatPromptBuilder(template=template))

rag_pipe.add_component(
    "llm",
    GoogleGenAIChatGenerator(
        api_key=Secret.from_token(config.GEMINI_API_KEY),
        model="gemini-2.5-flash-lite",  # Updated to match your indexing model version
    ),
)

# 4. Connect the dots
# 4. Connect the dots
rag_pipe.connect("embedder.embedding", "retriever.query_embedding")
rag_pipe.connect(
    "retriever.documents", "prompt_builder.documents"
)  # Explicitly use .documents
rag_pipe.connect(
    "prompt_builder.prompt", "llm.messages"
)  # prompt (list) -> messages (list)
rag_pipe.warm_up()


# 5. Run a test query
if __name__ == "__main__":
    while True:
        message = input("Enter a question: ")
        if message.strip().lower() in ("exit()", "quit()"):
            exit()

        query = message.strip()

        print(f"\nQuerying: {query}...")

        result = rag_pipe.run(
            data={"embedder": {"text": query}, "prompt_builder": {"question": query}},
            include_outputs_from=["retriever"],
        )

        print("\n--- RESPONSE ---")
        print(result["llm"]["replies"][0].text)
        print("\n--- SOURCES ---")
        for doc in result["retriever"]["documents"]:
            print(
                f"- {doc.meta.get('title', 'Unknown')}-{doc.meta.get('chunk_index', 'Unknown')} [score:{doc.score:.3f}]"
            )
        print("\n", "------" * 10, "\n")
