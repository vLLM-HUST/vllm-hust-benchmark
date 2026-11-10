def main():
    """Frozen screen/performance runner. Fresh engine per invocation; no output-based tuning."""
    import dataclasses
    import json
    import os
    import sys
    import time
    from pathlib import Path

    from scoring import extract_answer
    from vllm import LLM, SamplingParams

    root = Path(__file__).resolve().parent
    arm = sys.argv[1]
    assert arm in {"q35-mtp4", "q35-adaptive"}
    dataset = sys.argv[2]
    assert dataset in {"pilot", "screen280", "perf56"}
    batch = 16
    output_cap = 16 if len(sys.argv) > 3 and sys.argv[3] == "capacity" else 4000
    assert os.environ.get("HUST_VSPEC_ADAPTIVE_TRACE", "0") == "0"
    config = json.loads((root / "gamma4-configs.json").read_text())[arm]
    label = os.environ["LUMI_RUN_LABEL"]
    (root / "evidence" / f"{label}-config.json").write_text(
        json.dumps(
            {
                "engine": config,
                "dataset": dataset,
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": output_cap},
                "warmup": "14 pilot prompts, 16 output tokens",
            },
            indent=2,
        )
    )
    llm = LLM(**config)
    rows = [
        json.loads(line)
        for line in (root / f"{dataset}.jsonl").read_text().strip().split("\n")
    ]
    tokenizer = llm.get_tokenizer()
    prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        }
        for r in rows
    ]
    assert all(isinstance(t, int) for p in prompts for t in p["prompt_token_ids"])
    assert all(len(p["prompt_token_ids"]) + output_cap <= 16384 for p in prompts)
    label = os.environ["LUMI_RUN_LABEL"]

    def serial(x):
        if dataclasses.is_dataclass(x):
            return dataclasses.asdict(x)
        if hasattr(x, "__dict__"):
            return vars(x)
        return repr(x)

    warm_rows = [
        json.loads(line)
        for line in (root / "pilot.jsonl").read_text().strip().split("\n")
    ]
    warm_prompts = [
        {
            "prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        }
        for r in warm_rows
    ]
    llm.generate(
        warm_prompts, SamplingParams(temperature=0, max_tokens=16), use_tqdm=False
    )
    from partial_results import install as install_partial_results

    install_partial_results(
        llm.llm_engine,
        root / "evidence" / f"{label}-partial.jsonl",
        rows,
        prompts,
        serial,
        reject_degenerate_token0=True,
    )
    witnesses = {"before": llm.collective_rpc("gamma4_runtime_witness", timeout=30)}
    before = llm.get_metrics()
    start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, top_p=1, max_tokens=output_cap),
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - start
    result = []
    for r, p, o in zip(rows, prompts, outputs, strict=True):
        c = o.outputs[0]
        pred = extract_answer(c.text)
        result.append(
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "gold": r["answer"],
                "prediction": pred,
                "correct": pred == r["answer"],
                "invalid": pred is None,
                "input_tokens": len(p["prompt_token_ids"]),
                "output_tokens": list(c.token_ids),
                "text": c.text,
                "finish_reason": c.finish_reason,
                "metrics": o.metrics,
            }
        )
    witnesses["after"] = llm.collective_rpc("gamma4_runtime_witness", timeout=30)
    (root / "evidence" / f"{label}-witness.json").write_text(
        json.dumps(witnesses, indent=2)
    )
    record = {
        "arm": arm,
        "dataset": dataset,
        "batch": batch,
        "elapsed": elapsed,
        "rows": result,
        "metrics_before": before,
        "metrics_after": llm.get_metrics(),
    }
    (root / "evidence" / f"{label}-result.json").write_text(
        json.dumps(record, default=serial, indent=2)
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "elapsed": elapsed,
                "correct": sum(x["correct"] for x in result),
                "n": len(result),
                "output_tokens": sum(len(x["output_tokens"]) for x in result),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
