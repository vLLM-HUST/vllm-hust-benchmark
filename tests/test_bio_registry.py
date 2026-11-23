"""CPU contract fixtures: these are not model-family hardware acceptance."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vllm_hust_benchmark.bio_interfaces import DriverSession
from vllm_hust_benchmark.bio_registry import Registry, read_document

EMPTY = {"type": "object", "additionalProperties": False}


class Driver:
    api_version = 1
    config_schema = EMPTY
    opened = 0

    def supports(self, contract):
        return contract["input_type"] == "sequence_batch" and contract["version"] == 1

    def validate(self, binding, case, engine_options):
        if not case["sequences"]:
            raise ValueError("Empty logical batch")

    def open(self, profile, assets, engine_options):
        type(self).opened += 1
        return Runtime()

    def execute(self, runtime, binding, case, identity):
        return {"task": binding.operation, "identity": identity}

    def diagnostics(self, *args):
        return None

    def work(self, *args):
        return {"jobs": 1}

    def rejections(self, *args):
        return []


class Input(Driver):
    def prepare(self, binding, case, assets, upstream):
        return {"sequences": case["sequences"]}


class Reference(Driver):
    def load(self, *args):
        return None

    def expected(self, *args):
        return None


class Evaluator(Driver):
    def compare(self, *args):
        return [{"passed": True}]


class Collector:
    api_version = 1
    config_schema = EMPTY

    def snapshot(self, runtime, profile, reset=False):
        return {"reset": reset}


class Runtime:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class BrokenDriver(Driver):
    execute = None


class WrongSignature(Driver):
    def execute(self, one_argument_only):
        return None


class WrongAPI(Driver):
    api_version = 2


def document():
    classes = {
        "driver": Driver,
        "input": Input,
        "reference": Reference,
        "evaluator": Evaluator,
        "collector": Collector,
    }
    return {
        "schema_version": 1,
        "id": "cpu-fixture",
        "version": 1,
        "asset_manifest": "assets.json",
        "legacy_phase1_regression": False,
        "components": [
            {
                "id": kind,
                "version": 1,
                "kind": kind,
                "interface_version": 1,
                "entrypoint": __name__ + ":" + cls.__name__,
            }
            for kind, cls in classes.items()
        ],
        "contracts": [
            {
                "id": "sequence.embedding",
                "version": 1,
                "operation": "embedding",
                "input_type": "sequence_batch",
                "output_type": "token_embedding_batch",
                "parameters": EMPTY,
            }
        ],
        "profiles": [
            {
                "id": "fixture-model",
                "version": 1,
                "asset_id": "fixture-asset",
                **{kind: {"id": kind, "version": 1} for kind in classes},
                "limits": {
                    "max_context": 128,
                    "dtypes": ["float16"],
                    "enforce_eager": [True],
                    "input_alphabet": "AC",
                },
                "capabilities": {
                    "batch": True,
                    "streaming": False,
                    "cancel": False,
                    "concurrency": False,
                },
                "tasks": [
                    {
                        "id": "embedding",
                        "contract": {"id": "sequence.embedding", "version": 1},
                    }
                ],
                "rejection_probes": ["fixture_invalid"],
            }
        ],
        "gates": {"error": 0.01},
    }


CASE = {
    "profile_id": "fixture-model",
    "task_id": "embedding",
    "options": {},
    "sequences": ["AC"],
}
OPTIONS = {"dtype": "float16", "enforce_eager": True, "max_model_len": 128}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        Driver.opened = 0

    def test_dispatch_and_host_exception_close_are_owned(self):
        registry = Registry(self.doc)
        registry.preflight([CASE], OPTIONS)
        self.assertEqual(Driver.opened, 0)
        session = registry.session("fixture-model", Path("."), OPTIONS)
        with self.assertRaisesRegex(RuntimeError, "host failure"):
            with session:
                binding = registry.bind("fixture-model", "embedding")
                prepared = binding.prepare(CASE, None, None)
                self.assertEqual(
                    session.execute(binding, CASE, prepared)["task"], "embedding"
                )
                self.assertEqual(session.observe(reset=True), {"reset": True})
                raise RuntimeError("host failure")
        session.close()
        self.assertTrue(session.closed)
        self.assertEqual(session.runtime.close_calls, 1)
        with self.assertRaises(RuntimeError):
            session.execute(binding, CASE, prepared)
        with self.assertRaises(RuntimeError):
            session.start()

    def test_new_profile_reuses_task_contract_and_all_hooks(self):
        alias = deepcopy(self.doc["profiles"][0])
        alias["id"] = "another-release"
        self.doc["profiles"].append(alias)
        registry = Registry(self.doc)
        case = CASE | {"profile_id": alias["id"]}
        registry.preflight([case], OPTIONS)
        binding = registry.bind(alias["id"], "embedding")
        with registry.session(alias["id"], None, OPTIONS) as session:
            self.assertEqual(session.execute(binding, case, {})["task"], "embedding")
            with self.assertRaisesRegex(ValueError, "another runtime"):
                session.execute(registry.bind("fixture-model", "embedding"), CASE, {})

    def test_duplicate_ids_and_bindings_are_rejected(self):
        for key in ["components", "contracts", "profiles"]:
            value = deepcopy(self.doc)
            value[key].append(deepcopy(value[key][0]))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Duplicate"):
                Registry(value)
        self.doc["profiles"][0]["tasks"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate task"):
            Registry(self.doc)

    def test_unknown_kind_and_version_fail_before_allocation(self):
        for change, message in [
            ({"id": "absent"}, "Unknown"),
            ({"id": "input"}, "kind mismatch"),
            ({"version": 2}, "version"),
        ]:
            value = deepcopy(self.doc)
            value["profiles"][0]["driver"].update(change)
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, message),
            ):
                Registry(value)
        self.assertEqual(Driver.opened, 0)

    def test_unknown_profile_and_task(self):
        registry = Registry(self.doc)
        for profile, task in [("unknown", "embedding"), ("fixture-model", "unknown")]:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                registry.bind(profile, task)
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            registry.session("unknown", None, OPTIONS)

    def test_interface_version_and_noncallable_method(self):
        for name, exception in [
            ("WrongAPI", ValueError),
            ("BrokenDriver", TypeError),
            ("WrongSignature", TypeError),
        ]:
            value = deepcopy(self.doc)
            value["components"][0]["entrypoint"] = __name__ + ":" + name
            with self.subTest(name=name), self.assertRaises(exception):
                Registry(value).preflight([CASE], OPTIONS)
        self.assertEqual(Driver.opened, 0)

    def test_contract_signature_and_unsupported_contract_version(self):
        for changes in [{"input_type": "structure_batch"}, {"version": 2}]:
            value = deepcopy(self.doc)
            value["contracts"][0].update(changes)
            value["profiles"][0]["tasks"][0]["contract"]["version"] = value[
                "contracts"
            ][0]["version"]
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, "Incompatible task contract"),
            ):
                Registry(value).preflight([CASE], OPTIONS)

    def test_unknown_options_and_component_config_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "parameters"):
            Registry(self.doc).preflight([CASE | {"options": {"typo": 1}}], OPTIONS)
        self.doc["profiles"][0]["driver"]["config"] = {"typo": True}
        with self.assertRaisesRegex(ValueError, "configuration"):
            Registry(self.doc).preflight([CASE], OPTIONS)
        self.assertEqual(Driver.opened, 0)

    def test_execution_scope_and_batch_support(self):
        for changes in [
            {"dtype": "bfloat16"},
            {"enforce_eager": False},
            {"max_model_len": 129},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Registry(self.doc).preflight([CASE], OPTIONS | changes)
        self.doc["profiles"][0]["capabilities"]["batch"] = False
        with self.assertRaisesRegex(ValueError, "batches"):
            Registry(self.doc).preflight([CASE | {"sequences": ["AC", "CA"]}], OPTIONS)
        self.assertEqual(Driver.opened, 0)

    def test_json_duplicate_keys_nonfinite_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            for raw in ['{"id":1,"id":2}', '{"x":NaN}', '{"x":Infinity}']:
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    read_document(path)
        for name in ["../assets.json", "/tmp/assets.json"]:
            self.doc["asset_manifest"] = name
            with self.assertRaisesRegex(ValueError, "Invalid registry"):
                Registry(self.doc)

    def test_registry_identity_tracks_bound_configuration(self):
        registry = Registry(self.doc)
        before = registry.identity()
        registry.profiles["fixture-model"]["limits"]["max_context"] += 1
        self.assertNotEqual(registry.identity(), before)
        self.assertEqual(registry.document["profiles"][0]["limits"]["max_context"], 129)
        # Callers' input objects remain independent of the registry snapshot.
        self.assertEqual(self.doc["profiles"][0]["limits"]["max_context"], 128)

    def test_registry_identity_and_selected_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "registry.json").write_text(json.dumps(self.doc))
            (root / "assets.json").write_text(
                json.dumps(
                    {"profiles": {"fixture-asset": {"pinned": True}, "unused": {}}}
                )
            )
            registry = Registry.load(root / "registry.json")
            self.assertEqual(
                registry.asset_definitions(["fixture-model"]),
                {"profiles": {"fixture-asset": {"pinned": True}}},
            )
            self.assertIn(root / "registry.json", registry.source_files())
            before = registry.identity()
            changed = deepcopy(self.doc)
            changed["profiles"][0]["limits"]["max_context"] += 1
            self.assertNotEqual(Registry(changed).identity(), before)
            self.assertEqual(
                Registry(dict(reversed(list(self.doc.items())))).identity(), before
            )

    def test_close_failure_is_not_recorded_as_closed(self):
        registry = Registry(self.doc)
        session = registry.session("fixture-model", None, OPTIONS).start()
        with patch.object(
            session.runtime, "close", side_effect=RuntimeError("close failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                session.close()
            self.assertFalse(session.closed)
        session.close()

    def test_unstarted_session_and_empty_case_fail(self):
        session = DriverSession(
            Driver(), Collector(), self.doc["profiles"][0], None, OPTIONS
        )
        with self.assertRaises(RuntimeError):
            session.observe()
        with self.assertRaisesRegex(ValueError, "Empty"):
            Registry(self.doc).preflight([CASE | {"sequences": []}], OPTIONS)
        self.assertEqual(Driver.opened, 0)


class RegisteredReportTests(unittest.TestCase):
    def fixture(self, root):
        from test_bio_benchmark import BioBenchmarkTests, save
        from vllm_hust_benchmark.bio_report import read_json

        BioBenchmarkTests().build_fixture(root)
        value = document()
        profile = value["profiles"][0]
        profile["id"] = "esm2-150m"
        profile["tasks"][0]["id"] = "esm2.embedding"
        profile["rejection_probes"] = ["invalid" + str(i) for i in range(8)]
        registry = Registry(value)
        save(root / "registry.json", value)
        manifest = read_json(root / "manifest.json")
        manifest["registry"] = registry.identity()
        save(root / "manifest.json", manifest)
        for run in manifest["runs"]:
            path = root / run["directory"] / "native.json"
            native = read_json(path)
            native["registry"] = registry.identity()
            native["expected_rejections"] = profile["rejection_probes"]
            save(path, native)

    def test_report_uses_archived_probe_names_and_registry_identity(self):
        from test_bio_benchmark import save
        from vllm_hust_benchmark.bio_report import read_json, summarize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertEqual(summarize(root, numeric=False)["rejected_inputs"], 24)
            path = root / "start0/native.json"
            native = read_json(path)
            native["rejections"][0]["name"] = "unexpected_probe"
            save(path, native)
            with self.assertRaisesRegex(ValueError, "coverage"):
                summarize(root, numeric=False)

    def test_report_rejects_changed_registry_and_native_identity(self):
        from test_bio_benchmark import save
        from vllm_hust_benchmark.bio_report import read_json, summarize

        for target in ["registry", "native"]:
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.fixture(root)
                path = root / (
                    "registry.json" if target == "registry" else "start0/native.json"
                )
                value = read_json(path)
                if target == "registry":
                    value["version"] += 1
                else:
                    value["registry"]["sha256"] = "0" * 64
                save(path, value)
                with self.assertRaisesRegex(ValueError, "[Rr]egistry"):
                    summarize(root, numeric=False)

    def test_binding_from_changed_registry_cannot_use_session(self):
        registry = Registry(document())
        changed = document()
        changed["version"] += 1
        with registry.session("fixture-model", None, OPTIONS) as session:
            with self.assertRaisesRegex(ValueError, "registry identity"):
                session.execute(
                    Registry(changed).bind("fixture-model", "embedding"), CASE, {}
                )
