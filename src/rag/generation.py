"""Local-model boundary: stable prompt, strict output validation, raw retention."""

from __future__ import annotations

import json
from urllib import request

VALID_STANCES = {"supports", "refutes", "unknown"}
PROMPT = """You are an evidence-grounded QA component in a controlled benchmark.
Use only supplied documents. Repeated derivatives are not independent evidence. If a
document has source_root_conflict=true, answer unknown unless an independent primary
source resolves the conflict. Return exact JSON:
{{"stance":"supports|refutes|unknown","citations":["document_id"],"reason":"one concise sentence"}}

Question: {question}

Documents:
{documents}
"""


def validate_completion(
    parsed: object, allowed_ids: set[str]
) -> tuple[str, list[str], bool]:
    if not isinstance(parsed, dict):
        return "invalid", [], False
    stance, citations = parsed.get("stance"), parsed.get("citations")
    if (
        stance not in VALID_STANCES
        or not isinstance(citations, list)
        or not all(isinstance(value, str) for value in citations)
    ):
        return "invalid", [], False
    return stance, citations, set(citations).issubset(allowed_ids)


def llm_answer(
    question: str, documents: list[dict], model: str, endpoint: str, seed: int = 42
) -> dict:
    evidence = "\n\n".join(
        f"[document_id={doc['id']}; source_root={doc['root_id']}; role={doc['role']}; source_root_conflict={doc.get('root_conflict', False)}]\n{doc['text']}"
        for doc in documents
    )
    payload = {
        "model": model,
        "prompt": PROMPT.format(question=question, documents=evidence),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "seed": seed, "num_predict": 160},
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=180) as response:
        envelope = json.loads(response.read())
    raw = envelope.get("response", "")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "stance": "invalid",
            "citations": [],
            "reason": "Model did not return parseable JSON.",
        }
    metadata = {
        key: envelope.get(key)
        for key in (
            "model",
            "created_at",
            "total_duration",
            "prompt_eval_count",
            "eval_count",
        )
    }
    return {"raw": raw, "parsed": parsed, "ollama": metadata}
