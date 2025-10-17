# pinecone_upload.py
import json
import os
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer  # <-- Import this
from dotenv import load_dotenv

import config

load_dotenv()
# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

# INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
# VECTOR_DIM = int(os.environ["PINECONE_VECTOR_DIM"])  # Now 384

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = int(config.PINECONE_VECTOR_DIM)
# -----------------------------
# Initialize clients
# -----------------------------
# No longer need the OpenAI client
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=config.PINECONE_API_KEY)
# Load the free, local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,  # Will use 384 from your .env file
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)


# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts):
    """Generate embeddings using a local SentenceTransformer model."""
    return model.encode(texts).tolist()  # Use the local model


def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


# -----------------------------
# Main upload logic (no changes needed here)
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)

    print("All items uploaded successfully.")


if __name__ == "__main__":
    main()