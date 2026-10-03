import os

# ---- RAG target ----
OLLAMA_BASE      = "http://localhost:11434"
GEN_MODEL        = "mistral"
GEN_URL          = f"{OLLAMA_BASE}/api/generate"

# ---- Retrieval eval ----
TOP_K            = 5          # K for Recall@K / Precision@K / NDCG@K
RELEVANCE_LEVEL  = "page" # "chunk_id" | "page" | "filename"
# "page" matches on (filename, page), which is what we can label by hand from the PDF.

# ---- LLM-as-judge ----
JUDGE_MODEL      = "mistral"
JUDGE_TIMEOUT    = 120

# ---- Pricing (USD per 1M tokens) ----
# Local Ollama = 0. Add cloud models here when you swap them in.
PRICING = {
    "mistral":        {"input": 0.00,  "output": 0.00},
    "llama3":         {"input": 0.00,  "output": 0.00},
    "gpt-4o":         {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
}

# ---- Paths ----
RUNS_DIR = "eval_data/runs"
os.makedirs(RUNS_DIR, exist_ok=True)