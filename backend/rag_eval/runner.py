"""
                    GOLDEN DATASET
                         │
                         ▼
                 ┌───────────────┐
                 │   QUESTION    │
                 └───────┬───────┘
                         │
                         ▼
              ┌────────────────────┐
              │    RETRIEVAL       │
              │                    │
              │  RAGEngine.search  │
              └─────────┬──────────┘
                        │
                  Top-K chunks
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
      GOLDEN RELEVANCE        RETRIEVED RESULTS
             │                     │
             └──────────┬──────────┘
                        ▼
              ┌────────────────────┐
              │ RETRIEVAL METRICS  │
              │                    │
              │ Recall             │
              │ Precision          │
              │ Hit                │
              │ MRR                │
              │ NDCG               │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │    GENERATION      │
              │                    │
              │ Context + Question │
              │        ↓           │
              │       Mistral      │
              └─────────┬──────────┘
                        │
                        ▼
                  Generated Answer
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
    FAITHFULNESS    RELEVANCE    CORRECTNESS
          │             │             │
          │             │       Golden Answer
          │             │             │
          └─────────────┼─────────────┘
                        ▼
              ┌────────────────────┐
              │   SYSTEM METRICS   │
              │                    │
              │ Latency            │
              │ Tokens             │
              │ Throughput         │
              │ Cost               │
              └─────────┬──────────┘
                        │
                        ▼
                 EVALUATION REPORT
                        │
                        ▼
                 run_YYYYMMDD.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from . import config, dataset
from .metrics import (
    recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, hit_at_k,
    summarize_latencies, token_cost,
)
from .judge import faithfulness, answer_relevance, correctness
from .pipeline import run_retrieval, run_generation


def evaluate(dataset_path: str,
             top_k: int,
             sample: int = 0,
             skip_judge: bool = False) -> Dict[str, Any]:

    from rag_engine import RAGEngine
    engine = RAGEngine()

    items = dataset.load_dataset(dataset_path)
    if sample:
        items = items[:sample]

    per_item = []
    retrieval_latencies, generation_latencies = [], []
    in_tokens_total = out_tokens_total = 0
    cost_total = 0.0

    for item in items:
        qid = item["id"]
        question = item["question"]
        expected = item.get("expected_answer", "")

        # ---- retrieval ----
        # get all related chunks 
        hits, r_lat = run_retrieval(engine, question, top_k)
        retrieval_latencies.append(r_lat)

        retrieved_ids = [h["chunk_id"] for h in hits]
        
        # now check how many of the retrieved chunks are actually relevant, based on the labels in the dataset
        relevant_ids = dataset.relevant_id_set(
            item, hits, config.RELEVANCE_LEVEL
        )
        
        """
Recall@K => Did I retrieve the relevant chunks?
Relevant:
[A]
Retrieved top 5:
[A, B, C, D, E]
Recall@5 = 1 / 1 = 1.0 perfect if A was not in the top 5, recall would be 0.0

Precision@K = How much of what I retrieved was actually relevant?
Top 5:

A ← relevant
B
C
D
E

Precision@5 = 1/5 = 0.2 it means 1 is relevant out of 5 retrieved, if I had retrieved 2 relevant chunks, precision would be 2/5 = 0.4

Hit@K = Did I find at least one relevant result?
Top 5:

B
C
A ← relevant
D
E

Hit@5 = 1

MRR => How early did the first relevant result appear?
(This is particularly useful for RAG because you generally want useful context near the top.)

NDCD@K => How well did I rank the relevant results?
(How good is the ordering of relevant results within the top K?)



        """
        r_metrics = {
            f"recall@{top_k}":    recall_at_k(retrieved_ids, relevant_ids, top_k),
            f"precision@{top_k}": precision_at_k(retrieved_ids, relevant_ids, top_k),
            f"hit@{top_k}":       hit_at_k(retrieved_ids, relevant_ids, top_k),
            "mrr":                reciprocal_rank(retrieved_ids, relevant_ids),
            f"ndcg@{top_k}":      ndcg_at_k(retrieved_ids, relevant_ids, top_k),
        }

        # ---- generation ----
        gen = run_generation(question, hits)
        generation_latencies.append(gen["latency_s"])
        in_tokens_total  += gen["prompt_tokens"]
        out_tokens_total += gen["completion_tokens"]
        cost_total += token_cost(
            config.GEN_MODEL,
            gen["prompt_tokens"],
            gen["completion_tokens"],
            config.PRICING,
        )

        # scores are floats; reasons kept alongside for inspection
        g_scores = {"faithfulness": None, "relevance": None, "correctness": None}
        g_reasons = {}
        if not skip_judge:
            try:
                f = faithfulness(question, gen["context"], gen["answer"])
                g_scores["faithfulness"] = f["score"]
                g_reasons["faithfulness"] = f.get("reason", "")

                r = answer_relevance(question, gen["answer"])
                g_scores["relevance"] = r["score"]
                g_reasons["relevance"] = r.get("reason", "")

                if expected:
                    c = correctness(question, expected, gen["answer"])
                    g_scores["correctness"] = c["score"]
                    g_reasons["correctness"] = c.get("reason", "")
            except Exception as e:
                g_reasons["error"] = str(e)

        g_metrics = {**g_scores, "_reasons": g_reasons}

        per_item.append({
            "id": qid,
            "question": question,
            "answer": gen["answer"],
            "retrieved": [
                {"chunk_id": h["chunk_id"], "filename": h["filename"],
                 "page": h["page"], "rerank_score": h.get("rerank_score")}
                for h in hits
            ],
            "relevant_ids": sorted(relevant_ids),
            "retrieval_metrics": r_metrics,
            "generation_metrics": g_metrics,
            "system": {
                "retrieval_latency_s": r_lat,
                "generation_latency_s": gen["latency_s"],
                "prompt_tokens": gen["prompt_tokens"],
                "completion_tokens": gen["completion_tokens"],
            },
        })

    # ---- aggregate ----
    def avg(key, bucket="retrieval_metrics"):
        vals = [i[bucket][key] for i in per_item if i[bucket].get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    retrieval_summary = {
        k: avg(k) for k in
        [f"recall@{top_k}", f"precision@{top_k}", f"hit@{top_k}",
         "mrr", f"ndcg@{top_k}"]
    }

    generation_summary = {
        "faithfulness": avg("faithfulness", "generation_metrics"),
        "relevance":    avg("relevance",    "generation_metrics"),
        "correctness":  avg("correctness",  "generation_metrics"),
    }

    system_summary = {
        "retrieval_latency":  summarize_latencies(retrieval_latencies),
        "generation_latency": summarize_latencies(generation_latencies),
        "throughput_qps":     len(per_item) / max(1e-6, sum(generation_latencies)),
        "prompt_tokens_total":     in_tokens_total,
        "completion_tokens_total": out_tokens_total,
        "avg_prompt_tokens":       in_tokens_total  / max(1, len(per_item)),
        "avg_completion_tokens":   out_tokens_total / max(1, len(per_item)),
        "cost_usd_total":          round(cost_total, 6),
        "cost_usd_per_query":      round(cost_total / max(1, len(per_item)), 6),
    }

    return {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "gen_model": config.GEN_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "top_k": top_k,
            "relevance_level": config.RELEVANCE_LEVEL,
            "n_items": len(per_item),
        },
        "retrieval":  retrieval_summary,
        "generation": generation_summary,
        "system":     system_summary,
        "per_item":   per_item,
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _fmt(x):
    if x is None: return "  n/a"
    if isinstance(x, float): return f"{x:.3f}"
    return str(x)


def print_report(r: Dict):
    print("\n" + "=" * 62)
    print(f"RAG EVALUATION  —  {r['config']['n_items']} items  —  "
          f"model={r['config']['gen_model']}  top_k={r['config']['top_k']}")
    print("=" * 62)

    print("\nRETRIEVAL")
    for k, v in r["retrieval"].items():
        print(f"  {k:<18} {_fmt(v)}")

    print("\nGENERATION")
    for k, v in r["generation"].items():
        print(f"  {k:<18} {_fmt(v)}")

    print("\nSYSTEM")
    s = r["system"]
    for name in ("retrieval_latency", "generation_latency"):
        for k, v in s[name].items():
            print(f"  {name}.{k:<6}    {v*1000:.1f} ms")
    print(f"  throughput_qps     {s['throughput_qps']:.2f}")
    print(f"  prompt_tokens      {s['prompt_tokens_total']}")
    print(f"  completion_tokens  {s['completion_tokens_total']}")
    print(f"  cost_usd_total     ${s['cost_usd_total']:.6f}")
    print(f"  cost_usd_per_query ${s['cost_usd_per_query']:.6f}")
    print()


def main():
    p = argparse.ArgumentParser(description="RAG evaluation harness")
    p.add_argument("--dataset", default="eval_data/golden.example.json")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--sample", type=int, default=0,
                   help="only run the first N items (0 = all)")
    p.add_argument("--skip-judge", action="store_true",
                   help="skip LLM-as-judge (retrieval + system only)")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: eval_data/runs/run_<ts>.json)")
    args = p.parse_args()

    t0 = time.perf_counter()
    
    report = evaluate(args.dataset, args.top_k,
                      sample=args.sample, skip_judge=args.skip_judge)
    print(f"[eval] finished in {time.perf_counter() - t0:.1f}s")

    print_report(report)

    out_path = args.out or str(
        Path(config.RUNS_DIR) / f"run_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] full report → {out_path}")


if __name__ == "__main__":
    main()