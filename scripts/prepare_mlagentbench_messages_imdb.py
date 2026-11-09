"""Reuse the verified original IMDb asset binder with Messages-port verification."""

import argparse
import json
from pathlib import Path

import prepare_mlagentbench_imdb_offline as original_binding
from prepare_mlagentbench_messages_agent import verify_messages_port


def bind(args):
    previous = original_binding.verify_port
    try:
        original_binding.verify_port = verify_messages_port
        return original_binding.bind(args)
    finally:
        original_binding.verify_port = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--port-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--container-asset-root", type=Path, default=Path("/assets"))
    parser.add_argument("--container-cache-root", type=Path, default=Path("/hf"))
    print(json.dumps(bind(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
