"""Strict, versioned registration and binding; imports never create a runtime."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, resources
import inspect
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from vllm_hust_benchmark.bio_interfaces import DriverSession, INTERFACES


def read_document(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


@dataclass(frozen=True)
class TaskBinding:
    registry: "Registry"
    profile: dict
    declaration: dict
    contract: dict

    @property
    def operation(self):
        return self.contract["operation"]

    def component(self, kind):
        reference = self.declaration.get(kind, self.profile.get(kind))
        return self.registry.component(reference, kind)

    def prepare(self, case, assets, upstream):
        return self.component("input").prepare(self, case, assets, upstream)

    def compare(self, case, identity, actual, expected, diagnostics, gates):
        return self.component("evaluator").compare(
            self, case, identity, actual, expected, diagnostics, gates
        )


class Registry:
    def __init__(self, document, *, source=None):
        json.dumps(document, allow_nan=False)
        schema = json.loads(
            resources.files("vllm_hust_benchmark.data")
            .joinpath("bio_registry.schema.json")
            .read_text()
        )
        try:
            Draft202012Validator(schema).validate(document)
        except ValidationError as error:
            raise ValueError(f"Invalid registry: {error.message}") from error
        self.document = deepcopy(document)
        self.source = Path(source).resolve() if source else None
        self.components = self._index(self.document["components"], "component")
        self.contracts = self._index(self.document["contracts"], "contract")
        self.profiles = self._index(self.document["profiles"], "profile")
        self._loaded = {}
        self._bindings = {}
        for profile in self.profiles.values():
            all_ids = set()
            for kind in INTERFACES:
                self._check_reference(profile[kind], self.components, kind)
            for declaration in profile["tasks"]:
                task_id = declaration["id"]
                if task_id in all_ids:
                    raise ValueError(f"Duplicate task binding: {task_id}")
                all_ids.add(task_id)
                contract = self._check_reference(
                    declaration["contract"], self.contracts, "contract"
                )
                for kind in INTERFACES:
                    if kind in declaration:
                        self._check_reference(declaration[kind], self.components, kind)
                self._bindings[(profile["id"], task_id)] = TaskBinding(
                    self, profile, declaration, contract
                )

    @staticmethod
    def _index(rows, kind):
        result = {}
        for row in rows:
            if row["id"] in result:
                raise ValueError(f"Duplicate {kind} ID: {row['id']}")
            result[row["id"]] = row
        return result

    @staticmethod
    def _check_reference(reference, index, kind):
        row = index.get(reference["id"])
        if row is None:
            raise ValueError(f"Unknown {kind} binding: {reference['id']}")
        if row["version"] != reference["version"]:
            raise ValueError(f"Incompatible {kind} version: {reference['id']}")
        if kind != "contract" and row["kind"] != kind:
            raise ValueError(
                f"Component kind mismatch for {reference['id']}: expected {kind}"
            )
        return row

    @classmethod
    def load(cls, path):
        return cls(read_document(path), source=path)

    def bind(self, profile_id, task_id):
        try:
            return self._bindings[(profile_id, task_id)]
        except KeyError:
            raise ValueError(
                f"Unsupported publication/task: {profile_id}/{task_id}"
            ) from None

    def component(self, reference, kind):
        row = self._check_reference(reference, self.components, kind)
        key = row["id"]
        if key not in self._loaded:
            module, name = row["entrypoint"].split(":")
            implementation = getattr(import_module(module), name)
            instance = implementation()
            api = getattr(instance, "api_version", None)
            if type(api) is not int or api != row["interface_version"]:
                raise ValueError(f"Incompatible interface version: {key}")
            if not isinstance(instance, INTERFACES[kind]):
                raise TypeError(
                    f"Implementation does not satisfy {kind} interface: {key}"
                )
            for name, method in INTERFACES[kind].__dict__.items():
                if (
                    not name.startswith("_")
                    and callable(method)
                    and not callable(getattr(instance, name, None))
                ):
                    raise TypeError(f"Non-callable {kind} method: {key}.{name}")
                if not name.startswith("_") and callable(method):
                    signature = inspect.signature(getattr(instance, name))
                    arguments = [None] * (len(inspect.signature(method).parameters) - 1)
                    try:
                        if kind == "collector" and name == "snapshot":
                            signature.bind(None, None, reset=False)
                        else:
                            signature.bind(*arguments)
                    except TypeError as error:
                        raise TypeError(
                            f"Incompatible {kind} signature: {key}.{name}"
                        ) from error

            schema = instance.config_schema
            Draft202012Validator.check_schema(schema)
            self._loaded[key] = instance
        instance = self._loaded[key]
        try:
            Draft202012Validator(instance.config_schema).validate(
                reference.get("config", {})
            )
        except ValidationError as error:
            raise ValueError(
                f"Invalid {kind} configuration: {error.message}"
            ) from error
        return instance

    def preflight(self, cases, engine_options):
        """Resolve every requested hook and case before model/worker allocation."""
        for case in cases:
            binding = self.bind(case["profile_id"], case["task_id"])
            contract = binding.contract
            schema = contract["parameters"]
            Draft202012Validator.check_schema(schema)
            try:
                Draft202012Validator(schema).validate(case["options"])
            except ValidationError as error:
                raise ValueError(
                    f"Invalid task parameters or coordinates: {error.message}"
                ) from error
            limits = binding.profile["limits"]
            if engine_options["dtype"] not in limits["dtypes"]:
                raise ValueError("Execution precision is not supported by this profile")
            if engine_options["enforce_eager"] not in limits["enforce_eager"]:
                raise ValueError("Execution mode is not supported by this profile")
            if engine_options["max_model_len"] > limits["max_context"]:
                raise ValueError("Execution context exceeds the publication limit")
            if (
                len(case["sequences"]) > 1
                and not binding.profile["capabilities"]["batch"]
            ):
                raise ValueError("Profile does not support logical batches")
            for kind in INTERFACES:
                implementation = binding.component(kind)
                if kind != "collector" and not implementation.supports(contract):
                    raise ValueError(
                        f"Incompatible task contract for {kind}: {contract['id']}"
                    )
            binding.component("input").validate(binding, case, engine_options)
            binding.component("driver").validate(binding, case, engine_options)

    def session(self, profile_id, assets, engine_options):
        try:
            profile = self.profiles[profile_id]
        except KeyError:
            raise ValueError(f"Unknown profile: {profile_id}") from None
        return DriverSession(
            self.component(profile["driver"], "driver"),
            self.component(profile["collector"], "collector"),
            profile,
            assets,
            engine_options,
            self,
        )

    def source_files(self):
        files = set()
        if self.source:
            files.add(self.source)
        for row in self.components.values():
            module = import_module(row["entrypoint"].split(":")[0])
            source = inspect.getsourcefile(module)
            if source:
                files.add(Path(source).resolve())
        return sorted(files)

    def asset_definitions(self, profile_ids):
        if self.source is None:
            raise ValueError("Asset definitions require a file-backed registry")
        path = self.source.parent / self.document["asset_manifest"]
        definitions = read_document(path)["profiles"]
        keys = {self.profiles[name]["asset_id"] for name in profile_ids}
        if keys - definitions.keys():
            raise ValueError("A registered profile has no asset definition")
        return {"profiles": {key: definitions[key] for key in sorted(keys)}}

    def identity(self):
        raw = json.dumps(
            self.document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return {
            "id": self.document["id"],
            "version": self.document["version"],
            "sha256": sha256(raw).hexdigest(),
        }
