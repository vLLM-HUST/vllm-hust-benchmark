"""Per-arm divergence locus: natural (pre-EOS) vs post-EOS forced continuation.

Usage: ev2_natural_prefix.py A.json B.json [NAME]
A = OFF dump, B = ON dump (both list[list[int]] token ids).
An arm "diverges in the natural region" when its first differing step is before
the first EOS token of the OFF sequence (or the OFF sequence has no EOS).
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

MODEL = "/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct"


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else f"{a_path} vs {b_path}"
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = set(tok.encode("<|im_end|>") + tok.encode("<|endoftext|>"))
    A = json.load(open(a_path))
    B = json.load(open(b_path))

    def first_eos(seq):
        for i, t in enumerate(seq):
            if t in eos:
                return i
        return None

    rows = []
    nat = forced = 0
    for i, (x, y) in enumerate(zip(A, B)):
        if x == y:
            continue
        fd = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), None)
        fe = first_eos(x)
        within = (fe is None) or (fd is not None and fd < fe)
        nat += int(within)
        forced += int(not within)
        rows.append((i, fd, fe, "NATURAL" if within else "post-EOS"))

    print(f"[{name}] arms={len(A)} diverged={len(rows)} "
          f"natural_region={nat} post_EOS_forced={forced}")
    print("  arm  first_diff  first_EOS_in_OFF  locus")
    for r in rows:
        print(f"  {r[0]:>3}  {str(r[1]):>10}  {str(r[2]):>15}  {r[3]}")

    def cut(seq):
        fe = first_eos(seq)
        return seq[: fe + 1] if fe is not None else seq

    natural_prefix_match = sum(1 for x, y in zip(A, B) if cut(x) == cut(y))
    print(f"  natural-prefix exact match: {natural_prefix_match}/{len(A)}")


if __name__ == "__main__":
    main()
