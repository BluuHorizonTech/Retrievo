import json
import re
import requests
from typing import Dict

from . import config

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict:
    text = text.strip()
    # strip ``` fences
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = _JSON_RE.search(text)
    if not m:
        return {"score": 0.0, "reason": "judge returned no JSON"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"score": 0.0, "reason": "judge JSON parse failed"}


def _call_judge(prompt: str) -> Dict:
    r = requests.post(
        config.GEN_URL,
        json={
            "model": config.JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=config.JUDGE_TIMEOUT,
    )
    r.raise_for_status()
    raw = r.json().get("response", "")
    out = _extract_json(raw)
    try:
        out["score"] = max(0.0, min(1.0, float(out.get("score", 0.0))))
    except (TypeError, ValueError):
        out["score"] = 0.0
    out.setdefault("reason", "")
    return out


# ------------------------------------------------------------------
# FAITHFULNESS — every claim in the answer supported by context?
# ------------------------------------------------------------------
_FAITH_PROMPT = """You are a strict fact-checking judge.

CONTEXT (the only source of truth):
{context}

ANSWER (to evaluate):
{answer}

Task: Rate how FAITHFUL the answer is to the context.
- 1.0 = every claim is supported by the context
- 0.5 = some claims supported, some not
- 0.0 = answer contradicts or invents facts not in context

Respond ONLY with valid JSON:
{{"score": <float 0-1>, "reason": "<one sentence>"}}
"""


def faithfulness(question: str, context: str, answer: str) -> Dict:
    return _call_judge(_FAITH_PROMPT.format(context=context, answer=answer))


# ------------------------------------------------------------------
# ANSWER RELEVANCE — does the answer address the question?
# ------------------------------------------------------------------
_RELEVANCE_PROMPT = """You are a strict judge.

QUESTION:
{question}

ANSWER:
{answer}

Task: Rate how RELEVANT the answer is to the question.
- 1.0 = directly and completely answers it
- 0.5 = partially answers or is vague
- 0.0 = off-topic or non-responsive

Respond ONLY with valid JSON:
{{"score": <float 0-1>, "reason": "<one sentence>"}}
"""


def answer_relevance(question: str, answer: str) -> Dict:
    return _call_judge(_RELEVANCE_PROMPT.format(question=question, answer=answer))


# ------------------------------------------------------------------
# CORRECTNESS — does the answer match the reference answer?
# ------------------------------------------------------------------
_CORRECTNESS_PROMPT = """You are a strict grading judge.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{expected}

GENERATED ANSWER:
{answer}

Task: Rate how CORRECT the generated answer is compared to the reference.
- 1.0 = semantically equivalent to reference
- 0.5 = partially correct, missing or extra details
- 0.0 = wrong or contradicts the reference

Respond ONLY with valid JSON:
{{"score": <float 0-1>, "reason": "<one sentence>"}}
"""


def correctness(question: str, expected: str, answer: str) -> Dict:
    return _call_judge(
        _CORRECTNESS_PROMPT.format(question=question, expected=expected, answer=answer)
    )