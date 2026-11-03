"""Original QReCC questions -> model rewrite -> full-index BM25 -> model reader.

This declared local-LLM profile is not the paper's trained rewrite/reader baseline.
Original labels/rewrites are reserved for separate publisher scoring commands.
Default plan does not read data, open the index or contact a model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import traceback
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from probe_qrecc_original_index import JAR_SHA256
from score_qrecc_original_qa import TEST_SHA

ROOT = Path(__file__).resolve().parents[1]
SEARCH = """import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import io.anserini.search.SimpleSearcher;
class SearchOriginal {
 public static void main(String[] args) throws Exception {
  try (SimpleSearcher s = new SimpleSearcher(args[0])) {
   s.setBM25(0.82f, 0.68f);
   System.out.println("COUNT\\t" + s.getTotalNumDocuments());
   String q = Files.readString(Paths.get(args[1]), StandardCharsets.UTF_8);
   for (SimpleSearcher.Result h : s.search(q, Integer.parseInt(args[2]))) {
    if (h.raw == null) throw new IllegalStateException("stored original raw required");
    System.out.println("HIT\\t" + Base64.getEncoder().encodeToString(h.docid.getBytes(StandardCharsets.UTF_8)) + "\\t" + h.score + "\\t" + Base64.getEncoder().encodeToString(h.raw.getBytes(StandardCharsets.UTF_8)));
   }
  }
 }
}
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_inputs(data_file, start, count):
    if type(start) is not int or start < 0 or type(count) is not int or count < 1:
        raise ValueError("explicit nonnegative start and positive count required")
    if digest(data_file) != TEST_SHA:
        raise ValueError("unchanged pinned original QReCC test required")
    rows = json.loads(Path(data_file).read_bytes())
    if start + count > len(rows):
        raise ValueError("original selection exceeds test split")
    result = []
    for index in range(start, start + count):
        row = rows[index]
        if (
            not isinstance(row.get("Question"), str)
            or not isinstance(row.get("Context"), list)
            or any(not isinstance(x, str) for x in row["Context"])
            or any(type(row.get(k)) is not int for k in ["Conversation_no", "Turn_no"])
        ):
            raise ValueError("original question/history/identity contract required")
        result.append(
            {
                "original_row_index": index,
                **{
                    k: row[k]
                    for k in ["Conversation_no", "Turn_no", "Question", "Context"]
                },
            }
        )
    return result


def parse_hits(text, expected_documents):
    import base64
    import math

    counts = []
    hits = []
    identities = set()
    for line in text.splitlines():
        if line.startswith("COUNT\t"):
            counts.append(int(line.split("\t")[1]))
        elif line.startswith("HIT\t"):
            _, encoded_id, score, encoded_raw = line.split("\t")
            docid = base64.b64decode(encoded_id, validate=True).decode("utf-8")
            raw = json.loads(base64.b64decode(encoded_raw, validate=True))
            score = float(score)
            if (
                not docid
                or docid in identities
                or not math.isfinite(score)
                or raw.get("id") != docid
                or not isinstance(raw.get("contents"), str)
            ):
                raise ValueError("actual distinct stored original passages required")
            identities.add(docid)
            hits.append(
                {
                    "docid": docid,
                    "content": raw["contents"],
                    "score": score,
                    "rank": len(hits) + 1,
                }
            )
    if counts != [expected_documents]:
        raise ValueError("actual full original index document count differs")
    return hits


def model_call(*, endpoint, model, messages, budget, timeout, output):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": budget,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    (output / "request-private.json").write_text(json.dumps(body), encoding="utf-8")
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        status = response.code
    (output / "response-private.json").write_bytes(raw)
    (output / "http-status-private.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    if status != 200:
        raise RuntimeError("model HTTP failure retained; no automatic resubmission")
    result = json.loads(raw)
    content = result["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty original model output retained; no oracle fallback")
    return content.strip()


def run(
    *,
    data_file,
    index,
    java,
    jar,
    output,
    endpoint,
    model,
    start=0,
    count=3,
    retrieval_k=100,
    rewrite_budget=128,
    answer_budget=512,
    model_timeout=1800,
    heap_mib=2048,
    expected_documents=54241550,
):
    target = urlsplit(endpoint)
    if (
        target.scheme != "http"
        or target.hostname not in {"127.0.0.1", "localhost", "::1"}
        or target.username
        or target.password
    ):
        raise ValueError("owned loopback NPU model endpoint required")
    if any(
        type(x) is not int or x <= 0
        for x in [
            retrieval_k,
            rewrite_budget,
            answer_budget,
            model_timeout,
            heap_mib,
            expected_documents,
        ]
    ):
        raise ValueError("explicit positive profile limits required")
    tasks = selected_inputs(data_file, start, count)
    index = Path(index).resolve(strict=True)
    jar = Path(jar).resolve(strict=True)
    if (
        not index.is_dir()
        or not list(index.glob("segments_*"))
        or digest(jar) != JAR_SHA256
    ):
        raise ValueError("existing original Lucene index and pinned0.10 JAR required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("fresh private output required")
    output = output.resolve()
    if (
        output.exists()
        or output.is_relative_to(ROOT)
        or output.is_relative_to(index)
        or index.is_relative_to(output)
        or output.is_relative_to(Path(data_file).resolve().parent)
    ):
        raise ValueError("fresh private output separate from checkout/index required")
    version = subprocess.run(
        [str(java), "-version"], capture_output=True, text=True, check=True
    )
    if not re.search(r'(?:version\s+)?"11\.', version.stdout + version.stderr):
        raise ValueError("qualified original-profile Java11 required")
    output.mkdir(mode=0o700, parents=True)
    source = output / "SearchOriginal.java"
    source.write_text(SEARCH, encoding="utf-8")
    receipt = {
        "source_id": "QRECC",
        "status": "RUNNING_ORIGINAL_REWRITE_RETRIEVAL_READER",
        "protocol": "Declared local-LLM rewrite/read profile, original full-index BM25; not author trained baseline",
        "model": model,
        "endpoint": endpoint,
        "start": start,
        "count": count,
        "retrieval_k": retrieval_k,
        "bm25_k1": 0.82,
        "bm25_b": 0.68,
        "rewrite_budget": rewrite_budget,
        "answer_budget": answer_budget,
        "model_timeout_seconds": model_timeout,
        "expected_index_documents": expected_documents,
        "completed_original_positions": [],
        "failed_original_positions": [],
        "source_checkbox_eligible": False,
    }
    predictions = []
    rankings = []
    marker = output / "receipt-private.json"
    for task in tasks:
        identity = {
            k: task[k] for k in ["original_row_index", "Conversation_no", "Turn_no"]
        }
        directory = output / ("row-" + str(task["original_row_index"]))
        directory.mkdir(mode=0o700)
        (directory / "inputs-private.json").write_text(
            json.dumps(task), encoding="utf-8"
        )
        answer = ""
        hits = []
        try:
            stage = directory / "rewrite"
            stage.mkdir()
            rewrite = model_call(
                endpoint=endpoint,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the current conversational question into a standalone search query using only the supplied history and question. Return only the query.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {k: task[k] for k in ["Context", "Question"]},
                            ensure_ascii=False,
                        ),
                    },
                ],
                budget=rewrite_budget,
                timeout=model_timeout,
                output=stage,
            )
            query = directory / "query-private.txt"
            query.write_text(rewrite, encoding="utf-8")
            result = subprocess.run(
                [
                    str(java),
                    f"-Xmx{heap_mib}m",
                    "-XX:ActiveProcessorCount=2",
                    "--class-path",
                    str(jar),
                    str(source),
                    str(index),
                    str(query),
                    str(retrieval_k),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            (directory / "search-stdout-private.txt").write_text(
                result.stdout, encoding="utf-8"
            )
            (directory / "search-stderr-private.txt").write_text(
                result.stderr, encoding="utf-8"
            )
            if result.returncode:
                raise RuntimeError("original-index search failed; preserve logs")
            hits = parse_hits(result.stdout, expected_documents)
            if len(hits) > retrieval_k:
                raise ValueError("search returned more than declared original ranking")
            stage = directory / "reader"
            stage.mkdir()
            answer = model_call(
                endpoint=endpoint,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the supplied conversational question using the retrieved passages. Return only the answer; if unsupported, state that it is unknown.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "Question": task["Question"],
                                "Context": task["Context"],
                                "predicted_rewrite": rewrite,
                                "retrieved_passages": hits,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                budget=answer_budget,
                timeout=model_timeout,
                output=stage,
            )
            receipt["completed_original_positions"].append(task["original_row_index"])
        except Exception as error:
            (directory / "failure-private.json").write_text(
                json.dumps({"error_type": type(error).__name__}), encoding="utf-8"
            )
            (directory / "failure-traceback-private.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            receipt["failed_original_positions"].append(task["original_row_index"])
        predictions.append({**identity, "prediction": answer})
        rankings.append({**identity, "retrieved_passages": hits})
        marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (output / "predictions-private.json").write_text(
        json.dumps(predictions), encoding="utf-8"
    )
    (output / "rankings-private.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in rankings), encoding="utf-8"
    )
    if digest(data_file) != TEST_SHA:
        raise ValueError("original test changed during native execution")
    receipt["status"] = "ORIGINAL_NATIVE_ATTEMPTS_RETAINED_PUBLISHER_SCORING_REQUIRED"
    marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    for name in ["data-file", "index", "java", "jar", "output", "endpoint", "model"]:
        p.add_argument("--" + name)
    for name, default in [
        ("start", 0),
        ("count", 3),
        ("retrieval-k", 100),
        ("rewrite-budget", 128),
        ("answer-budget", 512),
        ("model-timeout", 1800),
        ("heap-mib", 2048),
        ("expected-documents", 54241550),
    ]:
        p.add_argument("--" + name, type=int, default=default)
    args = vars(p.parse_args())
    if not args.pop("execute"):
        print(
            json.dumps(
                {
                    "source_id": "QRECC",
                    "status": "PLAN_ONLY",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    if any(x is None for x in args.values()):
        p.error(
            "execution requires original data, index, Java11/JAR, private output and explicit local model"
        )
    result = run(**args)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "source_id",
                    "status",
                    "completed_original_positions",
                    "failed_original_positions",
                    "source_checkbox_eligible",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
