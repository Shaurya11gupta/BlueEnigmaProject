# Hybrid Knowledge AI System (Blue Enigma Labs Challenge)

This project is a sophisticated, hybrid Retrieval-Augmented Generation (RAG) system built for the Blue Enigma Labs AI Engineer Technical Challenge. It functions as an intelligent travel assistant for Vietnam by combining two different knowledge sources:

* **Neo4j (Graph Database):** Stores and queries structured, relational data (e.g., "Which *restaurant* is *located in* *Hanoi*?").
* **Pinecone (Vector Database):** Stores and searches unstructured text data based on semantic meaning (e.g., "What are some *romantic travel experiences*?").

The system uses a high-performance open-source model (Llama 3.1) via the free Groq API to reason over the combined context and generate high-quality, conversational answers.

## 🚀 Key Features & Advanced Improvements

This implementation goes beyond the basic requirements to create a more performant, cost-effective, and intelligent system:

* **Hybrid Retrieval:** Fetches context from both Pinecone and Neo4j to provide answers that are both factually accurate (from the graph) and contextually rich (from semantic search).
* **Cost-Effective & High-Performance LLM:** Uses the free, blazing-fast Groq API with the Llama 3.1 model for all reasoning and generation tasks, eliminating API costs.
* **Local, Private Embeddings:** All text embeddings are generated locally using the `sentence-transformers/all-MiniLM-L6-v2` model, ensuring data privacy and zero embedding costs.
* **Asynchronous Graph Queries:** The system uses `asyncio` and the `AsyncGraphDatabase` driver for Neo4j, preventing the application from blocking while waiting for database I/O.
* **Embedding Caching:** An in-memory cache (`embedding_cache`) stores computed embeddings, saving significant computation and time on repeated or similar queries.
* **Two-Step RAG (Summary + CoT):**
    1.  **Summarization:** A preliminary LLM call summarizes the "noisy" raw context from both databases into a dense, factual summary.
    2.  **Chain-of-Thought (CoT):** A final, powerful prompt instructs the LLM to "reason step-by-step" based on the summary *before* formulating its final answer, dramatically improving the quality and logic of the output.

## ⚙️ Architecture (Query Flow)

When a user asks a question, the system performs the following steps:

1.  **Embed Query:** The user's query is converted into a vector embedding using the local `sentence-transformers` model (checking the cache first).
2.  **Fetch Semantic Context:** The embedding is sent to **Pinecone** to find the `top_k` most semantically similar text chunks (locations, descriptions, etc.).
3.  **Fetch Graph Context:** The unique IDs from the Pinecone results are passed to **Neo4j**. An `async` Cypher query fetches all related, neighboring nodes and relationships (e.g., what's nearby, what city it's in).
4.  **Summarize Context:** The combined (and often large) context from Pinecone and Neo4j is sent to the LLM to be compressed into a concise summary.
5.  **Generate Answer:** The final prompt—containing the user's query, the concise summary, and a Chain-of-Thought instruction—is sent to the LLM (Groq) to generate a final, reasoned answer.

## 🛠️ Setup and Installation

Follow these steps to get the project running.

### 1. Prerequisites

* Python 3.9+
* A **Neo4j** instance (e.g., a free [AuraDB](https://neo4j.com/cloud/aura-graph-database/) cloud database)
* A **Pinecone** account (free tier)
* A **Groq** account (free tier)

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Set Up Environment Variables

Create a file named `.env` in the root of the project directory and fill it with your API keys and credentials.

**Use this template:**

```env
# --- Pinecone Configuration ---
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
# This MUST match the index name you create in the Pinecone dashboard
PINECONE_INDEX_NAME="vietnam-travel-index" 
# This MUST be 384 for the all-MiniLM-L6-v2 model
PINECONE_VECTOR_DIM=384 

# --- Neo4j Configuration ---
# Example for a local instance: bolt://localhost:7687
# Example for AuraDB: neo4j+s://xxxx.databases.neo4j.io
NEO4J_URI="YOUR_NEO4J_URI"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"

# --- Groq Configuration ---
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### 4. Install Dependencies

It is highly recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Activate it (macOS/Linux)
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

## 🚀 How to Run

You must load the data into the databases *before* you can run the chat.

### Step 1: Load Data into Neo4j

This script reads `vietnam_travel_dataset.json` and populates your Neo4j database with nodes and relationships.

```bash
python load_to_neo4j.py
```
*(Wait for this to complete)*

### Step 2: Load Data into Pinecone

This script reads the same JSON, generates embeddings locally, and uploads them to your Pinecone index.

```bash
python pinecone_upload.py
```
*(This will download the `all-MiniLM-L6-v2` model (approx. 90MB) on first run and may take a few minutes to process and upload all embeddings.)*

### Step 3: Run the Chat Assistant

Once both databases are populated, you can start the interactive chat.

```bash
python hybrid_chat.py
```

You can now ask complex questions like:
* "What are some romantic spots near the Hoan Kiem Lake?"
* "Create a 4-day itinerary for me that includes Hanoi and Hue."
* "Tell me about some good food to try in Ho Chi Minh City."
