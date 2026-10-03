import time
import requests
from typing import Dict, List, Tuple

from . import config


def run_retrieval(engine, question: str, top_k: int) -> Tuple[List[Dict], float]:
    t0 = time.perf_counter()
    hits = engine.retrieve(question, top_k=top_k)
    return hits, time.perf_counter() - t0


def build_prompt(question: str, hits: List[Dict], history: str = "") -> str:
    context = "\n\n".join(
        f"[{h['filename']} — page {h['page']}]\n{h['text']}" for h in hits
    )
    return f"""You are a helpful assistant.
Use the context below. Cite sources as (filename, page N).

Conversation so far:
{history}

Context:
{context}

User question:
{question}

Answer clearly."""


def run_generation(question: str, hits: List[Dict]) -> Dict:
    prompt = build_prompt(question, hits)
    t0 = time.perf_counter()
    r = requests.post(
        config.GEN_URL,
        json={
            "model": config.GEN_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    elapsed = time.perf_counter() - t0

    return {
        "answer": data.get("response", "").strip(),
        "prompt_tokens":     int(data.get("prompt_eval_count", 0)),
        "completion_tokens": int(data.get("eval_count", 0)),
        "latency_s": elapsed,
        "context": "\n\n".join(h["text"] for h in hits),
    }