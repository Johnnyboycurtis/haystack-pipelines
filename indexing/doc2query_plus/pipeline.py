import os
from typing import Any, Dict
from haystack import Pipeline
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

# Import your custom components
from .custom_components import LangChainSplitter, SyntheticQueryGenerator

SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_pipeline(
    gemini_api_key: str, pinecone_api_key: str, index_name: str = "demo"
):

    # 1. Define the Pinecone Spec (Serverless us-east-1 is default for free tier)
    pinecone_spec = {"serverless": {"region": "us-east-1", "cloud": "aws"}}

    # 2. Initialize Two Document Stores (One for each namespace)
    # This ensures Path A and Path B are physically separated in Pinecone

    fact_document_store = PineconeDocumentStore(
        api_key=Secret.from_token(pinecone_api_key),
        index=index_name,
        namespace="philosophy-doc2query",
        dimension=768,
        spec=pinecone_spec,
    )

    hook_document_store = PineconeDocumentStore(
        api_key=Secret.from_token(pinecone_api_key),
        index=index_name,
        namespace="philosophy-doc2query-synthetic",
        dimension=768,
        spec=pinecone_spec,
    )

    # 3. Initialize Pipeline
    indexing_pipeline = Pipeline()

    # --- Add Components ---

    # Shared Splitter
    indexing_pipeline.add_component(
        "splitter", LangChainSplitter(chunk_size=400, chunk_overlap=0)
    )

    # Path A: Factual Indexing
    indexing_pipeline.add_component(
        "fact_embedder",
        SentenceTransformersDocumentEmbedder(model=SENTENCE_TRANSFORMERS_MODEL),
    )
    indexing_pipeline.add_component(
        "fact_writer", DocumentWriter(document_store=fact_document_store)
    )

    # Path B: Synthetic Intent Indexing (Doc2Query++)
    indexing_pipeline.add_component(
        "synthetic_gen", SyntheticQueryGenerator(api_key=gemini_api_key)
    )
    indexing_pipeline.add_component(
        "hook_embedder",
        SentenceTransformersDocumentEmbedder(model=SENTENCE_TRANSFORMERS_MODEL),
    )
    indexing_pipeline.add_component(
        "hook_writer", DocumentWriter(document_store=hook_document_store)
    )

    # --- Connect Components ---

    # Path A (Facts)
    indexing_pipeline.connect("splitter.documents", "fact_embedder.documents")
    indexing_pipeline.connect("fact_embedder.documents", "fact_writer.documents")

    # Path B (Hooks)
    indexing_pipeline.connect("splitter.documents", "synthetic_gen.documents")
    indexing_pipeline.connect("synthetic_gen.hook_documents", "hook_embedder.documents")
    indexing_pipeline.connect("hook_embedder.documents", "hook_writer.documents")

    return indexing_pipeline
