# hybrid_chat.py
import os
import asyncio
from typing import List
from openai import OpenAI
from pinecone import Pinecone
from neo4j import AsyncGraphDatabase  # <-- FEATURE 3: Use async driver
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import config

load_dotenv()

# -----------------------------
# Config & Global Cache
# -----------------------------
GROQ_CHAT_MODEL = "llama-3.1-8b-instant"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# FEATURE 1: In-memory cache for embeddings
embedding_cache = {}

# -----------------------------
# Initialize clients & models
# -----------------------------
client = OpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key=config.GROQ_API_KEY,
)
pc = Pinecone(api_key=config.PINECONE_API_KEY)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ERROR: Index '{INDEX_NAME}' not found. Please run the pinecone_upload.py script first.")
    exit()
index = pc.Index(INDEX_NAME)

# FEATURE 3: Connect to Neo4j using the async driver
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
)


# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string, using a cache to avoid re-computation."""
    # FEATURE 1: Check cache first
    if text in embedding_cache:
        print("DEBUG: Embedding cache hit.")
        return embedding_cache[text]

    print("DEBUG: Embedding cache miss. Computing new embedding.")
    embedding = embedding_model.encode(text).tolist()
    embedding_cache[text] = embedding  # Store in cache
    return embedding


def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone. NOTE: Pinecone v3 lacks a native async client, so this remains synchronous."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Pinecone returned {len(res['matches'])} results.")
    return res["matches"]


async def fetch_graph_context_async(node_ids: List[str]):
    """FEATURE 3: Asynchronously fetch neighboring nodes from Neo4j."""
    facts = []
    cypher_query = """
    UNWIND $node_ids AS nid
    MATCH (n:Entity {id: nid})-[r]-(m:Entity)
    RETURN n.id AS source, type(r) AS rel, m.id AS target_id, m.name AS target_name, 
           left(m.description, 300) AS target_desc, labels(m) AS labels
    LIMIT 15
    """
    async with driver.session() as session:
        result = await session.run(cypher_query, node_ids=node_ids)
        # Correct way to consume the async iterator into a list
        facts = [record.data() async for record in result]

    print(f"DEBUG: Graph returned {len(facts)} facts asynchronously.")
    return facts


def summarize_context(user_query, pinecone_matches, graph_facts):
    """FEATURE 2: Use an LLM to summarize the retrieved context."""
    print("DEBUG: Summarizing retrieved context...")

    # Prepare the context in a readable format for the summarizer
    vec_context = "\n".join(
        [f"- {m['metadata'].get('name', '')}: {m['metadata'].get('text', '')[:200]}" for m in pinecone_matches])
    graph_context = "\n".join(
        [f"- The node '{f['source']}' is related to '{f['target_name']}' by the relationship '{f['rel']}'." for f in
         graph_facts])

    summary_prompt = f"""
    Based on the following search results and graph facts, briefly summarize the key entities and their relationships relevant to the user's query.
    Focus on creating a dense, factual summary.

    User Query: "{user_query}"

    Semantic Search Results:
    {vec_context}

    Graph Facts:
    {graph_context}

    Concise Summary:
    """

    resp = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=250,
        temperature=0.1
    )
    summary = resp.choices[0].message.content
    print(f"DEBUG: Generated Summary -> {summary}")
    return summary


def build_prompt_with_reasoning(user_query, summary):
    """FEATURE 4: Build a chat prompt that encourages Chain-of-Thought reasoning."""
    system = (
        "You are an expert travel assistant. Your goal is to create a helpful and coherent travel plan or answer. "
        "First, reason step-by-step about the user's query based on the provided summary of facts. "
        "After your reasoning, formulate a final, user-facing answer. "
        "Cite node ids like `[id:city_hanoi]` when referencing specific places."
    )

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
            f"User query: {user_query}\n\n"
            "Here is a summary of relevant information retrieved from our knowledge base:\n"
            f"{summary}\n\n"
            "Now, provide your step-by-step reasoning, and then the final answer for the user."
         }
    ]
    return prompt


def call_chat(prompt_messages):
    """Call the chat model."""
    resp = client.chat.completions.create(
        model=GROQ_CHAT_MODEL,
        messages=prompt_messages,
        max_tokens=600,
        temperature=0.3
    )
    return resp.choices[0].message.content


# -----------------------------
# Interactive chat
# -----------------------------
async def interactive_chat_async():
    """Main async chat loop."""
    print("Hybrid travel assistant (Advanced). Type 'exit' to quit.")
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        # Step 1: Synchronous Pinecone query
        matches = pinecone_query(query, top_k=TOP_K)

        # Step 2: Asynchronous Neo4j query
        match_ids = [m["id"] for m in matches]
        graph_facts = await fetch_graph_context_async(match_ids)

        # Step 3: Summarize context
        summary = summarize_context(query, matches, graph_facts)

        # Step 4: Build prompt with CoT and get final answer
        prompt = build_prompt_with_reasoning(query, summary)
        answer = call_chat(prompt)

        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")


if __name__ == "__main__":
    asyncio.run(interactive_chat_async())