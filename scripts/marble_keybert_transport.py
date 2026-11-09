"""Private queue transport for the original MARBLE KeyBERT call only.

Reuse the existing atomic queue writer; no prompts, neural computation,
network tools, Agent scheduling or grading are implemented in this module.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import uuid

from appworld_model_transport import save_new


class QueuedKeyBERT:
    def __init__(self, queue, *, timeout=600):
        path = Path(queue).absolute()
        if path.is_symlink() or not path.is_dir() or timeout <= 0:
            raise ValueError(
                "existing private keyword queue and positive timeout required"
            )
        self.queue = path
        self.timeout = timeout

    def extract_keywords(self, query, *, keyphrase_ngram_range, stop_words):
        if not isinstance(query, str):
            raise ValueError("original query string required")
        if keyphrase_ngram_range != (1, 3) or stop_words != "english":
            raise ValueError("original MARBLE keyword operation required")
        payload = json.dumps({"query": query}, ensure_ascii=False).encode()
        if len(payload) > 8 * 1024**2:
            raise ValueError("keyword transport bound exceeded; no truncation")
        identity = uuid.uuid4().hex
        save_new(self.queue / (identity + ".request.json"), payload)
        response = self.queue / (identity + ".response.json")
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "retained keyword request timed out; no resubmission"
                )
            time.sleep(0.1)
        if response.is_symlink() or response.stat().st_size > 8 * 1024**2:
            raise ValueError("invalid keyword response file")
        result = json.loads(response.read_bytes())
        if (
            not isinstance(result, dict)
            or result.get("status") != "KEYBERT_TOOL_RETURNED_NOT_AGENT_RESULT"
            or result.get("cpu_neural_fallback_used") is not False
            or result.get("model") != "sentence-transformers/all-MiniLM-L6-v2"
            or result.get("revision")
            != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # pragma: allowlist secret - pinned public model revision
            or not isinstance(result.get("keywords"), list)
        ):
            raise ValueError("qualified NPU keyword result required")
        keywords = []
        for row in result["keywords"]:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) not in (int, float)
                or not math.isfinite(row[1])
            ):
                raise ValueError("original keyword and finite score pairs required")
            keywords.append((row[0], row[1]))
        return keywords


@contextmanager
def keyword_transport(paper_collector, queue, *, timeout=600):
    """Bind only the original module's default KeyBERT factory, then restore it."""
    original = paper_collector.KeyBERT

    def factory():
        return QueuedKeyBERT(queue, timeout=timeout)

    paper_collector.KeyBERT = factory
    try:
        yield
    finally:
        paper_collector.KeyBERT = original
