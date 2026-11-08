"""Original-source workspace port controls; no publisher model execution."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "dailytalk_workspace", ROOT / "scripts/prepare_dailytalk_synthesis_workspace.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def test_plan_reads_no_data():
    assert ENTRY.prepare(SimpleNamespace(action="plan"))["model_calls"] == 0


def test_publisher_device_port_has_no_cpu_or_cuda_fallback():
    original = (
        "from g2p_en import G2p\n"
        "def english():\n    g2p = G2p()\n"
        "def run():\n"
        "    if torch.cuda.is_available():\n"
        '        torch.cuda.manual_seed(train_config["seed"])\n'
        "        device = torch.device('cuda')\n"
        "    else:\n        device = torch.device('cpu')\n"
    )
    result = ENTRY.port_source("synthesize.py", original.encode()).decode()
    compile(result, "ported.py", "exec")
    assert "torch.cuda" not in result
    assert 'device = torch.device(os.environ["DAILYTALK_NPU_DEVICE"])' in result
    assert 'torch.npu.manual_seed_all(train_config["seed"])' in result
    assert "    from g2p_en import G2p\n" in result
    with pytest.raises(ValueError, match="context"):
        ENTRY.port_source("synthesize.py", b"unexpected upstream code")


def test_optional_speaker_import_remains_in_both_original_branches():
    raw = (
        "from deepspeaker import embedding\n"
        "class Speaker:\n"
        "    def build(self):\n"
        '        if self.embedder_type == "DeepSpeaker":\n'
        "            return embedding.build_model()\n"
        "    def forward(self):\n"
        '        if self.embedder_type == "DeepSpeaker":\n'
        "            return embedding.predict_embedding()\n"
    ).encode()
    result = ENTRY.port_source("model/speaker_embedder.py", raw).decode()
    compile(result, "speaker.py", "exec")
    assert result.count("            from deepspeaker import embedding") == 2
    assert not result.startswith("from deepspeaker")
    assert ENTRY.port_source("unchanged.py", b"exact\r\n") == b"exact\r\n"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    args = SimpleNamespace(action="prepare", split="val", source_file=None)
    for name in (
        "author_root",
        "prepared",
        "metadata_root",
        "models_root",
        "audio_root",
        "encoder_output",
    ):
        setattr(args, name, tmp_path / name)
        getattr(args, name).mkdir()
    args.stage = tmp_path / "new-stage"
    files = {
        "config/DailyTalk/preprocess.yaml": yaml.safe_dump(
            {
                "path": {
                    "corpus_path": "old",
                    "raw_path": "old",
                    "lexicon_path": "old",
                    "preprocessed_path": "old",
                    "sub_dir_name": "data",
                },
                "preprocessing": {
                    "speaker_embedder": "none",
                    "audio": {"sampling_rate": 22050},
                },
            }
        ).encode(),
        "config/DailyTalk/train.yaml": yaml.safe_dump(
            {
                "path": {"ckpt_path": "old", "log_path": "old", "result_path": "old"},
                "seed": 1234,
            }
        ).encode(),
        "model/modules.py": b"# unmodified original fixture\r\n",
    }
    for name, raw in files.items():
        target = args.author_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    pins = tmp_path / "source.json"
    pins.write_text(
        json.dumps(
            {"revision": "fixture", "files": [{"path": name} for name in files]}
        ),
        encoding="utf-8",
    )
    profile = {}
    for role, name in (
        ("acoustic", "900000.pth.tar"),
        ("vocoder", "generator_universal.pth.tar"),
    ):
        raw = role.encode()
        (args.models_root / name).write_bytes(raw)
        profile[role] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    model_profile = tmp_path / "models.json"
    model_profile.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(ENTRY, "SOURCE_PROFILE", pins)
    monkeypatch.setattr(ENTRY, "PROFILE", model_profile)
    monkeypatch.setattr(ENTRY, "verify_source", lambda *_: None)
    monkeypatch.setattr(
        ENTRY, "load_inputs", lambda _: (["original"], {"utterances": []})
    )
    monkeypatch.setattr(ENTRY, "verify_saved", lambda *_: None)
    return args, files


def test_generated_configs_change_only_paths(fixture):
    args, files = fixture
    pins = json.loads(ENTRY.SOURCE_PROFILE.read_text("utf-8"))
    generated = ENTRY.build_files(args, pins)
    for relative in ("config/DailyTalk/train.yaml", "config/DailyTalk/preprocess.yaml"):
        before, after = (
            yaml.safe_load(files[relative]),
            yaml.safe_load(generated[relative]),
        )
        before.pop("path")
        after.pop("path")
        assert before == after
    assert generated["model/modules.py"] == files["model/modules.py"]


def test_fresh_workspace_refuses_overwrite_and_rechecks_tampering(fixture):
    args, _ = fixture
    try:
        result = ENTRY.prepare(args)
    except OSError as error:
        if sys.platform == "win32" and getattr(error, "winerror", None) == 1314:
            pytest.skip(
                "Windows requires symlink privilege; actual server preparation checked separately"
            )
        raise
    assert result["software_files"] == 3
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]
    with pytest.raises(FileExistsError):
        ENTRY.prepare(args)
    args.action = "verify"
    assert ENTRY.prepare(args)["original_utterances"] == 1
    (args.stage / "workspace/model/modules.py").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        ENTRY.prepare(args)


def test_asset_directory_overlap_rejected(fixture):
    args, _ = fixture
    args.stage = args.prepared / "unsafe"
    with pytest.raises(ValueError, match="overlap"):
        ENTRY.prepare(args)
