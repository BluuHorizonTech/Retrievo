<div align="center">

<img src="Retrievo.png" width="180"/>

# Retrievo

### Hybrid RAG for Document-Aware AI

[![Python](https://img.shields.io/badge/Python-3.x-blue)](...)
[![Flask](https://img.shields.io/badge/Flask-API-black)](...)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-white)](...)

</div>

# Retrievo

Retrievo lets you upload documents, index their content, retrieve relevant context using hybrid search, and generate answers with a local LLM through Ollama (you can also use other LLMs).  

It also includes a dedicated evaluation harness for measuring retrieval quality, answer quality, latency, token usage, throughput, and cost.

---

## Features

- PDF and text document ingestion
- Page-aware document processing
- Configurable text chunking with overlap
- Local embeddings through Ollama
- FAISS vector search
- Cosine-similarity retrieval using normalized embeddings
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- SQLite metadata and chunk storage
- Incremental document indexing
- SHA-256 document deduplication
- Background FAISS/BM25 index updates
- Streaming LLM responses
- Ollama-based local generation
- Session-based conversation memory
- Source-aware answers with filename and page information
- REST API through Flask
- RAG evaluation harness
- JSON evaluation reports
- Retrieval and generation performance metrics

---

## Architecture

```text
                         ┌─────────────────────┐
                         │       Frontend      │
                         │   Retrievo Web UI   │
                         └──────────┬──────────┘
                                    │
                              HTTP / REST
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Flask API       │
                         │                     │
                         │  /upload            │
                         │  /ask               │
                         │  /stats             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     RAG Engine      │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
               ┌────────┐      ┌────────┐      ┌────────────┐
               │ SQLite │      │ FAISS  │      │    BM25    │
               │        │      │        │      │            │
               │ chunks │      │ vectors│      │ lexical    │
               │ metadata│     │ search │      │ search     │
               └────────┘      └────────┘      └────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │ RRF Fusion     │
                           └───────┬────────┘
                                   │
                                   ▼
                           ┌────────────────┐
                           │ Cross Encoder  │
                           │   Reranker     │
                           └───────┬────────┘
                                   │
                                   ▼
                            Top-K Context
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │   Ollama LLM        │
                         │      Mistral        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                             Generated Answer
                                    │
                                    ▼
                              Streaming UI
```

---

## RAG Pipeline

### Document ingestion

```text
Document
   │
   ▼
Text extraction
   │
   ▼
Page-level text
   │
   ▼
Chunking
   │
   ▼
Embedding
   │
   ├──────────────► SQLite
   │                 metadata + chunks
   │
   └──────────────► FAISS
                     vectors + chunk IDs
```

### Query pipeline

```text
User Question
      │
      ▼
Query Embedding
      │
      ├──────────────► FAISS semantic search
      │
      └──────────────► BM25 lexical search
                              │
                              ▼
                       RRF score fusion
                              │
                              ▼
                       Candidate chunks
                              │
                              ▼
                       Cross-encoder
                         reranking
                              │
                              ▼
                         Top-K context
                              │
                              ▼
                         Ollama LLM
                              │
                              ▼
                       Generated answer
```

---

## Retrieval

Retrievo uses multiple retrieval stages.

### 1. Semantic retrieval

Embeddings are generated locally through Ollama.

The vector index uses:

```python
faiss.IndexFlatIP(VECTOR_DIM)
```

with L2-normalized vectors, allowing inner product search to represent cosine similarity.

FAISS IDs are mapped directly to SQLite `chunk_id` values through:

```python
faiss.IndexIDMap2
```

### 2. BM25 retrieval

BM25 provides lexical matching for exact terms, names, keywords, and phrases.

### 3. RRF fusion

Semantic and lexical rankings are combined using Reciprocal Rank Fusion.

```text
FAISS results
      +
BM25 results
      │
      ▼
   RRF fusion
      │
      ▼
Combined ranking
```

### 4. Cross-encoder reranking

The combined candidates are passed through a cross-encoder reranker before the final context is returned.

---

## Storage

### SQLite

SQLite is the source of truth for document and chunk metadata.

Main tables:

```text
documents
├── doc_id
├── filename
├── sha256
├── uploaded_at
└── num_chunks

chunks
├── chunk_id
├── doc_id
├── page
├── chunk_index
└── text
```

The `chunk_id` is also used as the corresponding FAISS vector ID.

### FAISS

FAISS stores the embedding vectors used for semantic retrieval.

The architecture keeps:

```text
SQLite chunk_id
       │
       │ same ID
       ▼
FAISS vector ID
```

This allows retrieved vector IDs to be resolved directly back to SQLite metadata and text.

---

## Document Processing

PDF documents are processed page by page.

Each extracted page is represented as:

```text
(page_number, text)
```

Text is then split into configurable word-based chunks.

Current defaults:

```text
Chunk size: 300 words
Overlap:    60 words
```

The overlap helps preserve context across neighboring chunks.

---

## Incremental Indexing

Retrievo avoids rebuilding the complete corpus for every upload.

When a new document is added:

```text
New document
     │
     ▼
SHA-256 check
     │
     ├── Already exists → skip
     │
     └── New document
             │
             ▼
          Chunk text
             │
             ▼
         Generate embeddings
             │
             ▼
       Store in SQLite
             │
             ▼
       Add vectors to FAISS
```

FAISS and BM25 updates use version tracking and background processing.

FAISS snapshots are written atomically using a temporary file followed by replacement.

---

## Local AI

Retrievo is designed around local inference.

Default Ollama endpoint:

```text
http://localhost:11434
```

Generation model:

```text
mistral
```

The embedding model and reranker are configurable in the RAG engine configuration.

---

## API

The Flask backend exposes the following endpoints.

### `GET /`

Health response:

```text
RAG Server Running 🚀
```

### `POST /upload`

Upload a PDF or UTF-8 text document.

Example:

```bash
curl -X POST \
  -F "file=@India.pdf" \
  http://localhost:5000/upload
```

Example response:

```json
{
  "message": "Document processed and stored in RAG"
}
```

### `POST /ask`

Ask a question against the indexed documents.

Request:

```json
{
  "question": "What is the capital of India?",
  "session_id": "default"
}
```

The response is streamed from the Ollama generation endpoint.

### `GET /stats`

Returns RAG engine statistics.

```bash
curl http://localhost:5000/stats
```

---

## Conversation Memory

The API maintains session-based chat history:

```python
chat_memory = {}
```

Each session stores user and assistant messages.

Current memory window:

```python
MEMORY_WINDOW = 10
```

The conversation history is included in the generation prompt.

Current implementation stores this memory in the Flask process, so it is not a persistent/distributed conversation store.

---

## Streaming Responses

The `/ask` endpoint requests streaming output from Ollama:

```python
"stream": True
```

The Flask server forwards generated chunks to the frontend as they arrive.

The UI can therefore display the answer progressively instead of waiting for the complete generation.

---

## Configuration

Important RAG configuration values include:

```text
Vector dimension:       768
Chunk size:             300
Chunk overlap:           60
Embedding batch size:    32
RRF K:                   60
```

The application also configures:

```text
Ollama generation endpoint
Embedding endpoint
Generation model
Embedding model
Reranker model
FAISS index path
SQLite database path
```

Refer to the project's configuration values for the exact runtime settings.

---

# Evaluation

Retrievo includes an evaluation harness under the `evaluation/` package.

The evaluation uses a golden JSON dataset containing:

```text
Question
Expected answer
Relevant document/page
```

Example:

```json
{
  "id": "q01",
  "question": "What is the official name of India?",
  "expected_answer": "Republic of India, also known as Bharat Ganarajya.",
  "relevant_chunks": [
    {
      "filename": "India - Wikipedia.pdf",
      "page": 1
    }
  ]
}
```

### Evaluation stages

```text
Golden Dataset
      │
      ▼
RAG Retrieval
      │
      ├── Recall@K
      ├── Precision@K
      ├── Hit@K
      ├── MRR
      └── NDCG@K
      │
      ▼
LLM Generation
      │
      ├── Faithfulness
      ├── Relevance
      └── Correctness
      │
      ▼
System Metrics
      │
      ├── Latency
      ├── Tokens
      ├── Throughput
      └── Cost
      │
      ▼
JSON Report
```

### Run evaluation

```bash
python -m evaluation.evaluate
```

Specific dataset:

```bash
python -m evaluation.evaluate \
  --dataset eval_data/golden.india.json
```

Custom Top-K:

```bash
python -m evaluation.evaluate \
  --dataset eval_data/golden.india.json \
  --top-k 5
```

Evaluate a sample:

```bash
python -m evaluation.evaluate \
  --dataset eval_data/golden.india.json \
  --sample 10
```

Skip LLM judge:

```bash
python -m evaluation.evaluate \
  --dataset eval_data/golden.india.json \
  --skip-judge
```

Specify report path:

```bash
python -m evaluation.evaluate \
  --dataset eval_data/golden.india.json \
  --out eval_data/runs/my_run.json
```

Evaluation reports are stored under:

```text
eval_data/runs/
```

---

## Evaluation Metrics

### Retrieval

```text
Recall@K
Precision@K
Hit@K
MRR
NDCG@K
```

### Generation

```text
Faithfulness
Answer Relevance
Correctness
```

### Performance

```text
Retrieval latency
Generation latency
P50
P95
Min
Max
Throughput QPS
Prompt tokens
Completion tokens
Total cost
Cost per query
```

Current retrieval evaluation uses:

```python
RELEVANCE_LEVEL = "page"
```

so manually labeled relevance is based on document filename and page.

---

## Example Evaluation Result

```text
==============================================================
RAG EVALUATION  —  3 items  —  model=mistral  top_k=5
==============================================================

RETRIEVAL
  recall@5           1.000
  precision@5        0.200
  hit@5              1.000
  mrr                0.567
  ndcg@5             0.673

GENERATION
  faithfulness       1.000
  relevance          1.000
  correctness        1.000

SYSTEM
  retrieval_latency.mean    ...
  generation_latency.mean   ...
  throughput_qps             ...
  prompt_tokens              ...
  completion_tokens          ...
  cost_usd_total             $...
  cost_usd_per_query         $...
```

---

# Project Structure

A typical repository layout:

```text
Retrievo/
│
├── app.py
├── rag_engine.py
├── config.py
│
├── evaluation/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── judge.py
│   ├── metrics.py
│   └── pipeline.py
│
├── eval_data/
│   ├── golden.example.json
│   └── runs/
│
├── uploads/
│
├── frontend/
│   └── ...
│
├── data/
│   └── ...
│
└── README.md
```

Adjust paths according to the actual repository layout.

---

# Setup

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Retrievo
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Ollama

```bash
ollama serve
```

Pull the required model:

```bash
ollama pull mistral
```

Also pull the embedding model configured by the RAG engine.

## 5. Start the backend

```bash
python app.py
```

The API runs on:

```text
http://localhost:5000
```

## 6. Start the frontend

If the frontend is served separately, run it through a local HTTP server instead of opening the HTML file directly.

Example:

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

---

# Runtime Components

| Component | Used For |
|---|---|
| Flask | REST API |
| SQLite | Document and chunk metadata |
| FAISS | Vector retrieval |
| BM25 | Lexical retrieval |
| RRF | Retrieval result fusion |
| Cross-Encoder | Reranking |
| Ollama | Local LLM and embeddings |
| PyPDF2 | PDF text extraction |
| Requests | HTTP communication |
| Python | Application runtime |

---

# Design Details

### Document identity

Documents are identified using SHA-256 hashes to prevent duplicate ingestion.

### Chunk identity

Every chunk receives a SQLite `chunk_id`.

The same ID is used when adding the vector to FAISS.

### Vector search

Vectors are L2-normalized before FAISS search.

```text
Normalized embeddings
        │
        ▼
IndexFlatIP
        │
        ▼
Cosine-equivalent similarity
```

### Retrieval combination

```text
Semantic ranking
       +
BM25 ranking
       │
       ▼
RRF
       │
       ▼
Cross-encoder reranking
       │
       ▼
Final context
```

---

# Data and Generated Files

Typical runtime files include:

```text
uploads/
```

Uploaded source documents.

```text
SQLite database
```

Document metadata and chunks.

```text
FAISS index
```

Vector embeddings and IDs.

```text
eval_data/runs/
```

Evaluation reports.

Runtime/generated data should generally not be committed to Git unless intentionally required.

---

# Current Limitations

- Conversation memory is process-local.
- Evaluation relevance is currently page-based by default.
- Evaluation runs are sequential.
- Local Ollama inference performance depends on the available hardware.
- PDF extraction quality depends on the PDF's text layer.
- The current API is intended for local/self-hosted usage.

---

