# Intent-Driven Document Expansion (Doc2Query++)

This project implements a **Doc2Query++** indexing pipeline using **Haystack 2.0**. It is designed to solve the "Lazy Prompter" problem in RAG systems by bridging the semantic gap between dense, factual documents and short, keyword-heavy user queries.

## 🎯 The Problem: The "Lazy Prompter"
In production RAG, retrieval often fails because:
*   **Users** provide fragments and shorthand (e.g., *"Spekit vs Highspot"*).
*   **Documents** are written in narrative, technical prose that lacks these shorthand "hooks."

This pipeline creates a **translation layer** by indexing not just what the document *is*, but what questions the document *answers*.

## 🏗️ Architecture
The pipeline implements a **Dual-Path Indexing Strategy**:

1.  **Path A (The Facts):** The original document is chunked and embedded.
2.  **Path B (The Hooks):** An LLM (Gemini 2.5 Flash) analyzes each chunk to predict 1-3 "Lazy Queries" a human might use to find that info. These queries are embedded as separate "Hook" documents.

Both paths are logically linked via `parent_chunk_id` and `document_id` metadata, allowing for seamless de-duplication and "swapping" during the retrieval phase.

## 📂 Project Structure
```text
indexing/
├── local_config.py          # API Keys and Index configurations (Gitignored)
├── demo_doc2query.py        # Entry point: Loads data and executes the pipeline
└── doc2query_plus/
    ├── __init__.py
    ├── custom_components.py # Custom Haystack components (Splitter & Generator)
    └── pipeline.py          # Haystack 2.0 Pipeline definition
```

## 🛠️ Components

### 1. `LangChainSplitter`
A custom Haystack component wrapping LangChain's `RecursiveCharacterTextSplitter`. It uses `tiktoken` encoding to ensure chunks stay within model context limits while preserving document lineage.

### 2. `SyntheticQueryGenerator`
The "Brain" of the pipeline. It uses **Google Gemini 2.5 Flash Lite** with Pydantic structured outputs to:
*   Generate 1-3 lazy queries per chunk.
*   Apply a "Self-Correction" filter (only indexing queries where `confidence` is `True`).
*   Attach the `original_text` to the hook's metadata for high-precision retrieval.

### 3. `PineconeDocumentStore`
Data is stored in **Pinecone** using two distinct namespaces:
*   `philosophy-doc2query`: Contains the factual text chunks.
*   `philosophy-doc2query-synthetic`: Contains the generated intent hooks.

## 🚀 Getting Started

### 1. Prerequisites
```bash
pip install haystack-ai google-genai pinecone-haystack langchain-text-splitters datasets tqdm
```

### 2. Configuration
Create a `local_config.py` in the `indexing/` directory:
```python
GEMINI_API_KEY = "your-google-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-index-name"
```

### 3. Run the Discovery
Execute the demo script to ingest the Stanford Encyclopedia of Philosophy dataset:
```bash
python demo_doc2query.py
```

## 📊 Technical Details
*   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
*   **LLM:** `gemini-2.5-flash-lite`
*   **Vector Dimension:** 768 (Note: Ensure your Pinecone index matches the embedding model dimension).
*   **Chunk Size:** 400 tokens.

## 📝 Note
This is an **experimental discovery project**. It is intended for demonstration and learning purposes in local environments. The pipeline is designed to be "immature" and flexible for rapid iteration on retrieval strategies.
EOF