"""Read the original Lucene index document count; default plan has no effects.

Uses trusted static Java source and the pinned original Anserini JAR. It does
not create/rebuild an index, execute model code or prove corpus provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
JAR_SHA256 = "b2f40e9dc56091364814e4146515d34a7a05af2be2a1cba5c88e3d25a50276f6"  # pragma: allowlist secret - observed original Pyserini0.10.0.1 archive member digest
COUNTER = """import java.nio.file.Paths;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.index.DirectoryReader;
class CountDocuments {
  public static void main(String[] args) throws Exception {
    try (FSDirectory dir = FSDirectory.open(Paths.get(args[0]));
         DirectoryReader reader = DirectoryReader.open(dir)) {
      System.out.println(reader.numDocs());
    }
  }
}
"""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for block in iter(lambda: inp.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def probe(*, index, java, jar, output, expected_documents=None, heap_mib=1024):
    if type(heap_mib) is not int or heap_mib < 1:
        raise ValueError("positive explicit Java heap budget required")
    if expected_documents is not None and (
        type(expected_documents) is not int or expected_documents < 0
    ):
        raise ValueError("nonnegative expected document count required")
    index = Path(index).resolve(strict=True)
    jar = Path(jar)
    if not index.is_dir() or not list(index.glob("segments_*")):
        raise ValueError("existing Lucene index required; never create it here")
    if jar.is_symlink() or not jar.is_file() or digest(jar) != JAR_SHA256:
        raise ValueError("unchanged original Pyserini0.10.0.1 Anserini JAR required")
    jar = jar.resolve()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("fresh private output required")
    output = output.resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(index):
        raise ValueError("probe output must be outside checkout and index")
    version = subprocess.run(
        [str(java), "-version"], capture_output=True, text=True, check=True
    )
    if not re.search(r'(?:version\s+)?"11\.', version.stdout + version.stderr):
        raise ValueError("qualified original-profile Java11 required")
    output.mkdir(mode=0o700, parents=True)
    source = output / "CountDocuments.java"
    source.write_text(COUNTER, encoding="utf-8")
    receipt = {
        "source_id": "QRECC",
        "status": "PROBING_ORIGINAL_INDEX_READONLY",
        "java_version": (version.stdout + version.stderr).strip(),
        "anserini_jar_sha256": JAR_SHA256,
        "counter_source_sha256": digest(source),
        "expected_documents": expected_documents,
        "model_calls": 0,
        "native_tasks_executed": 0,
        "corpus_provenance_verified": False,
        "source_checkbox_eligible": False,
    }
    marker = output / "receipt-private.json"
    try:
        result = subprocess.run(
            [
                str(java),
                f"-Xmx{heap_mib}m",
                "--class-path",
                str(jar),
                str(source),
                str(index),
            ],
            capture_output=True,
            text=True,
        )
        (output / "counter-stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output / "counter-stderr.txt").write_text(result.stderr, encoding="utf-8")
        receipt["counter_exit_code"] = result.returncode
        if result.returncode or not result.stdout.strip().isdigit():
            raise ValueError("original Lucene document count not returned; retain logs")
        count = int(result.stdout.strip())
        receipt["observed_documents"] = count
        if expected_documents is not None and expected_documents != count:
            raise ValueError(
                "actual Lucene document count differs from expected collection"
            )
        receipt["status"] = "ORIGINAL_INDEX_DOCUMENT_COUNT_VERIFIED_NOT_NATIVE_QA"
    except Exception as error:
        receipt.update(
            status="FAILED_INDEX_PROBE_RETAINED", error_type=type(error).__name__
        )
        marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise
    marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--java", type=Path, required=True)
    p.add_argument("--jar", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-documents", type=int)
    p.add_argument("--heap-mib", type=int, default=1024)
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
    print(
        json.dumps(
            probe(
                index=a.index,
                java=a.java,
                jar=a.jar,
                output=a.output,
                expected_documents=a.expected_documents,
                heap_mib=a.heap_mib,
            )
        )
    )


if __name__ == "__main__":
    main()
