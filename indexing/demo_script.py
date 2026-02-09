import pandas as pd

# from .pipeline import build_pipeline
from haystack import Document

# Load Dataset
import os

data_cache = "data/stanford_encyclopedia_of_philosophy.csv"

if os.path.isfile(data_cache):
    df = pd.read_csv(data_cache)
else:
    # df = pd.read_csv("hf://datasets/johnnyboycurtis/stanford_encyclopedia_of_philosophy/stanford_encyclopedia_of_philosophy.csv")
    from datasets import load_dataset

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("johnnyboycurtis/stanford_encyclopedia_of_philosophy")
    df = ds["train"].to_pandas()
    df.to_csv(data_cache, index=False)


# Convert DataFrame to Haystack Documents
raw_docs = [
    Document(content=row["contents"], meta={"title": row["entries"]})
    for _, row in df.head(10).iterrows()  # Testing with first 10 entries
]

indexing_pipeline = build_pipeline()
# Run Indexing
# Note: Because the LLM step is per-document, for large datasets
# you should run this in a loop or use Haystack's batch features.
for doc in raw_docs:
    indexing_pipeline.run({"splitter": {"documents": [doc]}})

print("Indexing Complete. Factual chunks and Synthetic Hooks are now in Pinecone.")
