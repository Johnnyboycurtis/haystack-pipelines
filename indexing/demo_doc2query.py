import os
import pandas as pd
from haystack import Document

# Import your pipeline builder
# Assuming your file is named pipeline.py
from doc2query_plus.pipeline import build_pipeline
from uuid import uuid4

# 1. Setup Environment and Data Cache
import local_config as config

GEMINI_API_KEY = config.GEMINI_API_KEY
PINECONE_API_KEY = config.PINECONE_API_KEY
PINECONE_INDEX_NAME = config.PINECONE_INDEX_NAME


data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_cache = os.path.join(data_dir, "stanford_encyclopedia_of_philosophy.csv")

# 2. Load Dataset
print("Checking for cached data...")
if os.path.isfile(data_cache):
    print("Loading from cache...")
    df = pd.read_csv(data_cache)
else:
    print("Downloading dataset from Hugging Face...")
    from datasets import load_dataset

    ds = load_dataset("johnnyboycurtis/stanford_encyclopedia_of_philosophy")
    df = ds["train"].to_pandas()
    # generate document IDs; standard in a structured database
    df["document_id"] = [str(uuid4()) for _ in range(df.shape[0])]
    df.to_csv(data_cache, index=False)

# 3. Prepare Documents
print("Preparing documents for indexing...")
# We'll take the first 10 entries for this experiment
raw_docs = []
for _, row in df.head(1).iterrows():
    doc = Document(
        id=row["document_id"], content=row["contents"], meta={"title": row["entry"]}
    )
    raw_docs.append(doc)


# 4. Initialize the Pipeline
print("Building the Doc2Query++ Pipeline...")
indexing_pipeline = build_pipeline(
    gemini_api_key=GEMINI_API_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    index_name=PINECONE_INDEX_NAME,
)

print(indexing_pipeline)

# 5. Run Indexing
# We pass the entire list. The splitter will chunk them,
# and the generator will process each chunk.
print(f"Starting indexing for {len(raw_docs)} source documents...")
output = indexing_pipeline.run({"splitter": {"documents": raw_docs}})
print("\n" + "=" * 30)
print("INDEXING COMPLETE")
print("Path A: Factual chunks are in namespace 'factual-index'")
print("Path B: Synthetic Hooks are in namespace 'synthetic-index'")
print("=" * 30)
