"""CPU fixtures only; not XL-Sum model-generation or scoring evidence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from vllm_hust_workload_data import xlsum
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


spec = importlib.util.spec_from_file_location(
    "xlsum_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_xlsum_original.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row():
    return {
        "id": "same-id",
        "url": "https://example.test",
        "title": "title",
        "text": " original\narticle ",
        "summary": "reference only",
    }


def test_all_variants_are_pinned_original_archive_pairs():
    source = source_spec("XLSUM")
    archives = {f["path"]: f for f in source["files"] if f["role"] == "source_archive"}
    tests = [f for f in source["files"] if f["role"] == "data"]
    assert len(archives) == len(tests) == 45
    assert len({f["language_variant"] for f in tests}) == 45
    for test in tests:
        archive = archives[test["archive_path"]]
        assert archive["language_variant"] == test["language_variant"]
        assert test["rows"] >= 500
        assert test["path"].endswith("_test.jsonl")


def test_summary_is_reference_not_input_and_all_original_fields_survive():
    original = row()
    record = adapt_row(
        "XLSUM",
        original,
        source={"file_sha256": "fixture"},
        spec=adapter_spec("XLSUM"),
        index=0,
    )
    assert record.task["input"] == {"text": original["text"]}
    assert record.outcome == {"summary": original["summary"]}
    assert record.replay["annotations"] == {
        k: original[k] for k in ["id", "title", "url"]
    }
    assert record.replay["unmapped_field_names"] == []
    assert original["text"] == " original\narticle "


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop("summary"),
        lambda r: r.update(extra="unregistered"),
        lambda r: r.update(text=""),
        lambda r: r.update(id=3),
    ],
)
def test_invalid_or_drifted_source_fails_without_silent_field_loss(change):
    data = row()
    change(data)
    with pytest.raises(ValueError):
        list(xlsum.validate_rows([data]))


def test_duplicate_id_within_variant_fails():
    with pytest.raises(ValueError, match="duplicate"):
        list(xlsum.validate_rows([row(), row()]))


def fixture_archive(tmp_path, monkeypatch, bad=False):
    payload = (json.dumps(row(), ensure_ascii=False) + "\n").encode()
    archive = tmp_path / "data/lang_XLSum_v2.0.tar.bz2"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:bz2") as stream:
        member = tarfile.TarInfo("./lang_test.jsonl")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    source = {
        "revision": "fixture",
        "access": {"license": ["cc-by-nc-sa-4.0"]},
        "files": [
            {
                "path": "data/lang_XLSum_v2.0.tar.bz2",
                "role": "source_archive",
                "bytes": archive.stat().st_size,
                "sha256": runner.sha256(archive),
            },
            {
                "path": "lang/lang_test.jsonl",
                "role": "data",
                "archive_path": "data/lang_XLSum_v2.0.tar.bz2",
                "archive_member": "./lang_test.jsonl",
                "bytes": len(payload),
                "sha256": "wrong" if bad else hashlib.sha256(payload).hexdigest(),
                "rows": 1,
            },
        ],
    }
    monkeypatch.setattr(xlsum, "source_spec", lambda sid: source)
    return payload


def test_exact_original_test_member_bytes_not_reconstructed(tmp_path, monkeypatch):
    payload = fixture_archive(tmp_path, monkeypatch)
    output = tmp_path / "fresh"
    receipt = xlsum.prepare_original_archives(archive_root=tmp_path, output=output)
    assert (output / "lang/lang_test.jsonl").read_bytes() == payload
    assert receipt["original_test_rows"] == 1
    assert receipt["native_generation_or_scoring_verified"] is False
    with pytest.raises(ValueError, match="fresh"):
        xlsum.prepare_original_archives(archive_root=tmp_path, output=output)


def test_corrupt_member_does_not_create_output(tmp_path, monkeypatch):
    fixture_archive(tmp_path, monkeypatch, bad=True)
    with pytest.raises(ValueError, match="identity"):
        xlsum.prepare_original_archives(
            archive_root=tmp_path, output=tmp_path / "fresh"
        )
    assert not (tmp_path / "fresh").exists()


def test_changed_original_scorer_is_rejected_before_import(tmp_path):
    file = tmp_path / "code.py"
    file.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="identity"):
        runner.original_scoring(
            tmp_path,
            {
                "source_files": [
                    {"path": "code.py", "bytes": file.stat().st_size, "sha256": "wrong"}
                ]
            },
        )


def test_modelcard_normalization_does_not_mutate_raw_article():
    data = row()
    assert runner.normalized_article(data["text"]) == "original article"
    assert data["text"] == " original\narticle "
    assert runner.PROTOCOL["encoder_max_length"] == 512
    assert runner.PROTOCOL["decoder_max_length"] == 84
    assert runner.PROTOCOL["num_beams"] == 4
    assert runner.PROTOCOL["length_penalty"] == 0.6
    assert runner.PROTOCOL["tie_word_embeddings"] is False


def test_incompatible_preflight_is_not_reused(tmp_path):
    (tmp_path / "receipt-public.json").write_text(
        json.dumps(
            {
                "status": "PUBLISHER_MODEL_CARD_NPU_GENERATION_COMPLETED_SCORING_PENDING",
                "language_variant": "english",
                "model_revision": "fixture",
                "author_tie_word_embeddings": True,
            }
        )
    )
    with pytest.raises(ValueError, match="incompatible"):
        runner.preflight_cache(
            tmp_path,
            row=row(),
            offset=0,
            tokenizer=SimpleNamespace(),
            pins={"model_revision": "fixture"},
        )


def test_native_cache_exact_inputs_outputs_and_private_hash_required(tmp_path):
    data = row()
    original = {
        "original_row": data,
        "publisher_normalized_article": "original article",
        "encoder_input_ids": [[1, 2]],
        "generated_token_ids": [0, 3, 1],
        "summary": "generated",
    }
    private = tmp_path / "english-0.private.json"
    private.write_text(json.dumps(original))
    receipt = {
        "source_id": "XLSUM",
        "source_revision": "source",
        "data_revision": "data",
        "model_revision": "model",
        "protocol": runner.PROTOCOL,
        "versions": {"transformers": "4.57.3"},
        "languages": [
            {
                "language_variant": "english",
                "data_sha256": "data-hash",
                "samples": [
                    {
                        "original_offset": 0,
                        "private_record_sha256": runner.sha256(private),
                    }
                ],
            }
        ],
    }
    (tmp_path / "receipt-public.json").write_text(json.dumps(receipt))

    class Tokenizer:
        def __call__(self, inputs, **kwargs):
            assert inputs == ["original article"]
            assert kwargs == {
                "padding": "max_length",
                "truncation": True,
                "max_length": 512,
            }
            return {"input_ids": [[1, 2]]}

        def decode(self, ids, **kwargs):
            assert ids == [0, 3, 1]
            assert kwargs == {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
            }
            return "generated"

    options = {
        "asset": {"language_variant": "english", "sha256": "data-hash"},
        "row": data,
        "offset": 0,
        "tokenizer": Tokenizer(),
        "pins": {"source_revision": "source", "model_revision": "model"},
        "data_revision": "data",
    }
    assert runner.native_cache(tmp_path, **options) == original
    with pytest.raises(ValueError, match="original request"):
        runner.native_cache(
            tmp_path, **{**options, "row": {**data, "text": "different article"}}
        )
    private.write_text("changed")
    with pytest.raises(ValueError, match="private evidence changed"):
        runner.native_cache(tmp_path, **options)


def test_prepare_cli_uses_same_original_extractor(tmp_path, monkeypatch, capsys):
    from vllm_hust_benchmark.workload_data import main

    fixture_archive(tmp_path, monkeypatch)
    assert (
        main(
            [
                "prepare-xlsum-original",
                "--archive-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "fresh"),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["original_test_rows"] == 1
    assert not receipt["native_generation_or_scoring_verified"]


def test_original_tokenizer_uses_isolated_model_free_worker(monkeypatch):
    process = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO(
            "\n".join(
                [
                    json.dumps(
                        {"transformers": "4.11.3", "tokenizer_class": "T5Tokenizer"}
                    ),
                    json.dumps({"input_ids": [[1, 2]]}),
                    json.dumps({"text": "original decode"}),
                ]
            )
            + "\n"
        ),
        poll=lambda: 0,
    )

    def launch(command, **kwargs):
        assert command[1:3] == ["-I", "-u"]
        assert command[4:] == ["--tokenizer-worker", "--model-root", "model"]
        assert "shell" not in kwargs
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    tokenizer = runner.OriginalTokenizer("python", "model")
    assert tokenizer(
        ["fixture"], padding="max_length", truncation=True, max_length=512
    ) == {"input_ids": [[1, 2]]}
    assert (
        tokenizer.decode(
            [1, 2], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        == "original decode"
    )
    sent = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
    assert [message["method"] for message in sent] == ["encode", "decode"]
    assert sent[0]["inputs"] == ["fixture"]
    tokenizer.close()
