"""Run the unchanged original QReCC retrieval scorer on saved ranked passages.

Default plan only. Reference answers are joined exclusively in private scoring
input, never in the saved retrieval contract. This does not prove retrieval or
model provenance, build an index, or qualify end-to-end native delivery.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_jsonschemabench_author_source import verify_source
from score_qrecc_original_qa import TEST_SHA

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/qrecc-author-source.json"
WORKER = """
import importlib.metadata,json,multiprocessing,pathlib,sys
if importlib.metadata.version('nltk') != '3.5':
    raise RuntimeError('original nltk==3.5 isolated scorer client required')
root=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'utils'))
import evaluate_retrieval as author
identities=json.loads(pathlib.Path('query-identities.private.json').read_text())
groups={str(x['Conversation-ID'])+'_'+str(x['Turn-ID']):[] for x in identities}
with open('evaluation-input.private.jsonl') as inp:
    with multiprocessing.Pool(int(sys.argv[2])) as pool:
        for doc in pool.imap(author.compute_f1_for_retrieved_passage,inp):
            groups[str(doc['Conversation-ID'])+'_'+str(doc['Turn-ID'])].append(doc)
empty=sum(not docs for docs in groups.values())
print('Total number of unique queries: '+str(len(groups)))
print('Empty retrieval queries retained: '+str(empty))
print('Mean Reciprocal Rank (MRR): '+format(author.compute_mean_reciprocal_rank(groups,author.RELEVANCE_THRESHOLD),'.4f'))
print('Recall@10: '+format(author.compute_recall_at_k(groups,10,author.RELEVANCE_THRESHOLD)*100,'.2f')+'%')
print('Recall@100: '+format(author.compute_recall_at_k(groups,100,author.RELEVANCE_THRESHOLD)*100,'.2f')+'%')
temp=pathlib.Path('author-score-private');temp.mkdir()
upper=author.compute_extractive_upper_bounds(groups,temp) if not empty else (None,None)
for label,value in zip(['EM','F1'],upper):
    print('Extractive Upper Bound for '+label+' (100 point scale): '+('undefined' if value is None else format(value*100,'.2f')))
"""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for block in iter(lambda: inp.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def convert_rankings(rows, rankings, output):
    """Preserve supplied ranks; add labels only in an evaluation-only file."""
    selected = []
    with (
        Path(rankings).open(encoding="utf-8") as inp,
        Path(output).open("x", encoding="utf-8") as out,
    ):
        for line in inp:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {
                "original_row_index",
                "Conversation_no",
                "Turn_no",
                "retrieved_passages",
            }:
                raise ValueError(
                    "saved identity and ranked passages only; no reference fields"
                )
            index = item["original_row_index"]
            if (
                type(index) is not int
                or not 0 <= index < len(rows)
                or (selected and index <= selected[-1])
            ):
                raise ValueError(
                    "distinct original row indices in original order required"
                )
            row = rows[index]
            if (
                type(item["Conversation_no"]) is not int
                or type(item["Turn_no"]) is not int
                or any(item[k] != row[k] for k in ["Conversation_no", "Turn_no"])
                or not isinstance(row.get("Answer"), str)
            ):
                raise ValueError(
                    "original conversation, turn and answer identity required"
                )
            passages = item["retrieved_passages"]
            if not isinstance(passages, list) or not 0 <= len(passages) <= 100:
                raise ValueError(
                    "retain zero to100 actual hits; never omit or fabricate empty retrieval"
                )
            ids = set()
            for rank, passage in enumerate(passages, 1):
                if not isinstance(passage, dict) or set(passage) != {
                    "docid",
                    "content",
                    "rank",
                    "score",
                }:
                    raise ValueError("original retrieved passage fields only")
                if (
                    not isinstance(passage["docid"], str)
                    or not passage["docid"]
                    or passage["docid"] in ids
                    or not isinstance(passage["content"], str)
                    or type(passage["rank"]) is not int
                    or passage["rank"] != rank
                    or type(passage["score"]) not in [int, float]
                    or not math.isfinite(passage["score"])
                ):
                    raise ValueError(
                        "unique document IDs, contiguous original ranks and finite scores required"
                    )
                ids.add(passage["docid"])
                out.write(
                    json.dumps(
                        {
                            "Conversation-ID": row["Conversation_no"],
                            "Turn-ID": row["Turn_no"],
                            "docid": passage["docid"],
                            "content": passage["content"],
                            "answer": row["Answer"],
                            "rank": rank,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            selected.append(index)
    if not selected:
        raise ValueError("no preselected original tasks; do not report empty scoring")
    return selected


def score(
    *, data_file, author_root, rankings_file, output, python=sys.executable, workers=1
):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("one to8 original scorer workers required")
    author_root = verify_source(Path(author_root), PINS)
    inputs = [Path(data_file), Path(rankings_file)]
    hashes = [digest(p) for p in inputs]
    if hashes[0] != TEST_SHA:
        raise ValueError("unchanged pinned original QReCC test required")
    rows = json.loads(inputs[0].read_text("utf-8"))
    output = Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(ROOT):
        raise ValueError("fresh private scoring output outside checkout required")
    output = output.resolve()
    output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "QRECC",
        "status": "PREPARING_SAVED_RETRIEVAL_SCORING",
        "original_test_sha256": hashes[0],
        "saved_rankings_sha256": hashes[1],
        "original_author_profile_sha256": digest(PINS),
        "model_calls": 0,
        "retrieval_provenance_verified": False,
        "generation_provenance_verified": False,
        "source_checkbox_eligible": False,
    }
    marker = output / "receipt-private.json"
    try:
        selected = convert_rankings(
            rows, inputs[1], output / "evaluation-input.private.jsonl"
        )
        (output / "query-identities.private.json").write_text(
            json.dumps(
                [
                    {
                        "Conversation-ID": rows[i]["Conversation_no"],
                        "Turn-ID": rows[i]["Turn_no"],
                    }
                    for i in selected
                ]
            ),
            encoding="utf-8",
        )
        receipt["selected_original_rows"] = selected
        process = subprocess.run(
            [str(python), "-IB", "-c", WORKER, str(author_root), str(workers)],
            cwd=output,
            capture_output=True,
            text=True,
        )
        (output / "author-stdout.txt").write_text(process.stdout, encoding="utf-8")
        (output / "author-stderr.txt").write_text(process.stderr, encoding="utf-8")
        receipt["author_exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError(
                "original retrieval scorer failed; retain logs and evaluation input"
            )
        count = re.search(
            r"(?m)^Total number of unique queries: (\d+)$", process.stdout
        )
        if not count or int(count[1]) != len(selected):
            raise ValueError("author query count differs; no silently omitted tasks")
        empty = re.search(
            r"(?m)^Empty retrieval queries retained: (\d+)$", process.stdout
        )
        if not empty or not 0 <= int(empty[1]) <= len(selected):
            raise ValueError("missing explicit empty retrieval accounting")
        # Original functions define metrics; retain the author's display precision.
        metrics = {}
        for label in [
            "Mean Reciprocal Rank (MRR)",
            "Recall@10",
            "Recall@100",
            "Extractive Upper Bound for EM (100 point scale)",
            "Extractive Upper Bound for F1 (100 point scale)",
        ]:
            match = re.search(
                r"(?m)^" + re.escape(label) + r": ([0-9.]+%?|undefined)$",
                process.stdout,
            )
            if not match:
                raise ValueError("missing original function metric")
            if match[1] == "undefined" and (
                not label.startswith("Extractive Upper Bound") or int(empty[1]) == 0
            ):
                raise ValueError(
                    "undefined metric outside original upper-bound empty-hit gap"
                )
            metrics[label] = None if match[1] == "undefined" else match[1]
        if [digest(p) for p in inputs] != hashes:
            raise ValueError(
                "original test or saved ranking input changed during scoring"
            )
        verify_source(author_root, PINS)
        receipt.update(
            status="SAVED_RANKINGS_SCORED_BY_UNCHANGED_AUTHOR_FUNCTIONS_NOT_NATIVE_DELIVERY",
            queries=len(selected),
            empty_retrieval_queries=int(empty[1]),
            author_function_api_not_unmodified_cli=True,
            author_reported_metrics=metrics,
            extractive_upper_bounds_are_not_model_answers=True,
        )
    except Exception as error:
        receipt.update(
            status="FAILED_SCORING_RETAINED_NOT_NATIVE_DELIVERY",
            error_type=type(error).__name__,
        )
        marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise
    marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--author-root", type=Path, required=True)
    p.add_argument("--rankings-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
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
    result = score(
        data_file=a.data_file,
        author_root=a.author_root,
        rankings_file=a.rankings_file,
        output=a.output,
        python=a.python,
        workers=a.workers,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "source_id",
                    "status",
                    "queries",
                    "model_calls",
                    "source_checkbox_eligible",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
