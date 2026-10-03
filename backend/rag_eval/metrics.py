import math
import statistics
from typing import List, Dict, Set

# ------------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------------
def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for x in retrieved[:k] if x in relevant) / k


def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    for i, x in enumerate(retrieved, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def _dcg(rels: List[float]) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    rels = [1.0 if x in relevant else 0.0 for x in retrieved[:k]]
    dcg  = _dcg(rels)
    ideal_len = min(len(relevant), k)
    idcg = _dcg([1.0] * ideal_len)
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    return 1.0 if (set(retrieved[:k]) & relevant) else 0.0


# ------------------------------------------------------------------
# SYSTEM
# ------------------------------------------------------------------
def summarize_latencies(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
    s = sorted(values)
    def pct(p):
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]
    return {
        "mean": statistics.mean(s),
        "p50":  statistics.median(s),
        "p95":  pct(0.95),
        "min":  s[0],
        "max":  s[-1],
    }


def token_cost(model: str, in_tokens: int, out_tokens: int, pricing: Dict) -> float:
    p = pricing.get(model, {"input": 0.0, "output": 0.0})
    return (in_tokens * p["input"] + out_tokens * p["output"]) / 1_000_000