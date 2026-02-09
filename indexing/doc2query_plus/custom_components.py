import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types
from haystack import component, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


@component
class LangChainSplitter:
    """Custom component using LangChain's RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 0):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        all_chunks = []
        for doc in documents:
            chunks = self.splitter.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                metadata = doc.meta.copy()
                metadata["document_id"] = doc.id
                metadata["chunk_index"] = i
                curr_chunk = Document(id=f"{doc.id}_{i}", content=chunk, meta=metadata)
                all_chunks.append(curr_chunk)
        return {"documents": all_chunks}


# --- Pydantic Models for Gemini Structured Output ---


class LazyQuery(BaseModel):
    text: str = Field(
        description="A short, fragmented, or keyword-heavy search query a human might type."
    )
    confidence: bool = Field(
        description="True if the synthetic lazy query is accurately answered by the text, False otherwise."
    )


class LazyQueryResponse(BaseModel):
    queries: List[LazyQuery] = Field(
        description="A list of predicted user queries.", min_length=1, max_length=5
    )


def json_parser(raw_text: str) -> Optional[dict]:
    """Extracts JSON from markdown blocks if necessary and parses into Pydantic model."""
    try:
        # Remove markdown code blocks if present
        clean_json = re.sub(r"```json\s?|\s?```", "", raw_text).strip()
        data = json.loads(clean_json)
        return data
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Manual JSON parsing failed: {e}")
        return None


prompt_template = """
You are an expert at predicting user search intent. 
Given the following factual document chunk, generate 1-3 "Lazy Queries". 
These are short, keyword-heavy, or poorly phrased fragments a human would type if they were in a hurry.

Rules:
1. Only generate queries that are directly answered by the text.
2. Assign a confidence score (0.0 to 1.0) based on how well the query matches the text.
3. Return ONLY a JSON object.

Text: {text}
"""


@component
class SyntheticQueryGenerator:
    """
    Generates synthetic 'Lazy Queries' using Google Gemini.
    Filters by confidence and preserves metadata linkage for de-duplication.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        prompt_template: str = prompt_template,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
        self.prompt_template = prompt_template

        self.system_instruction = (
            "You are an expert at predicting 'Lazy' user search intent. "
            "Users are often in a hurry and type fragments or keywords rather than full questions. "
            "Generate 1 to 3 queries that represent how a user would find the provided text. "
            "Only include queries that are explicitly supported by the facts in the text. "
            "Example: Instead of 'Compare Spekit and Highspot and their products' --> 'Spekit vs Highspot'"
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        queries = []
        synthetic_documents = []
        print("Generating Synthetic Queries")
        for doc in tqdm(documents):
            try:
                message = self.prompt_template.format(text=doc.content)
                parsed_response = self._generate(message=message)

                if not parsed_response:
                    continue

                for q in parsed_response.queries:
                    queries.append(q.model_dump())
                    # Self-Correction Filter: Only keep hooks where confidence is True
                    if q.confidence:
                        synthetic_documents.append(
                            Document(
                                content=q.text,
                                meta={
                                    "parent_chunk_id": doc.id,
                                    "document_id": doc.meta[
                                        "document_id"
                                    ],  # original parent document
                                    "lazy_query": q.text,
                                    "original_text": doc.content,  # Critical for inference swap
                                    "intent_type": "lazy_query",
                                    "source_title": doc.meta["title"],
                                },
                            )
                        )
            except Exception as e:
                print(f"Error processing doc {doc.id}: {e}")
                continue

        return {
            "documents": synthetic_documents,
            "metadata": {"queries": queries},
        }

    def _generate(self, message: str, attempts: int = 3) -> Optional[LazyQueryResponse]:
        """Handles the API call to Gemini with retries and fallback parsing."""
        for i in range(1, attempts + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=message,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        response_mime_type="application/json",
                        response_schema=LazyQueryResponse,
                        temperature=0.7,
                    ),
                )

                # Case 1: SDK successfully parsed the response
                if response.parsed:
                    return response.parsed

                # Case 2: SDK failed to parse, but we have text
                if response.text:
                    data = json_parser(response.text)  # returns Dictionary
                    if data:
                        manual_parsed = LazyQueryResponse(**data)
                        return manual_parsed

            except Exception as e:
                print(f"Attempt {i} failed for Gemini API: {e}")
                if i < attempts:
                    time.sleep(2 * i)  # Exponential backoff

        return None
