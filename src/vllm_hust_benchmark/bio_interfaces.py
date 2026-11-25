"""Version-1 structural interfaces for native biological benchmark integration.

Implementations may live in the model repository. No interface owns model math.
Prepared input, native output and reference output keep their task-specific types.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RuntimeDriver(Protocol):
    def supports(self, contract) -> bool: ...

    api_version: int
    config_schema: dict

    def validate(self, binding, case, engine_options) -> None: ...
    def open(self, profile, assets, engine_options) -> Any: ...
    def execute(self, runtime, binding, case, identity) -> Any: ...
    def diagnostics(self, runtime, binding, case, identity, output) -> Any: ...
    def work(self, binding, case, identity, output) -> dict: ...
    def rejections(self, runtime, profile, identity) -> list: ...


@runtime_checkable
class InputAdapter(Protocol):
    def supports(self, contract) -> bool: ...

    api_version: int
    config_schema: dict

    def validate(self, binding, case, engine_options) -> None: ...
    def prepare(self, binding, case, assets, upstream) -> dict: ...


@runtime_checkable
class ReferenceProvider(Protocol):
    def supports(self, contract) -> bool: ...

    api_version: int
    config_schema: dict

    def load(self, profile, assets, upstream) -> Any: ...
    def expected(self, state, binding, case, identity, actual) -> Any: ...


@runtime_checkable
class OutputEvaluator(Protocol):
    def supports(self, contract) -> bool: ...

    api_version: int
    config_schema: dict

    def compare(
        self, binding, case, identity, actual, expected, diagnostics, gates
    ) -> list: ...


@runtime_checkable
class ResourceCollector(Protocol):
    api_version: int
    config_schema: dict

    def snapshot(self, runtime, profile, reset=False) -> dict: ...


INTERFACES = {
    "driver": RuntimeDriver,
    "input": InputAdapter,
    "reference": ReferenceProvider,
    "evaluator": OutputEvaluator,
    "collector": ResourceCollector,
}


class DriverSession:
    """Own exactly one native runtime, including the host-exception close path."""

    def __init__(
        self, driver, collector, profile, assets, engine_options, registry=None
    ):
        self.driver, self.collector, self.profile = driver, collector, profile
        self.assets, self.engine_options = assets, dict(engine_options)
        self.registry = registry
        self.runtime = None
        self.closed = False

    def start(self):
        if self.runtime is not None or self.closed:
            raise RuntimeError("A driver session can only be started once")
        self.runtime = self.driver.open(self.profile, self.assets, self.engine_options)
        if not callable(getattr(self.runtime, "close", None)):
            raise TypeError("Native runtime must provide close()")
        return self

    def execute(self, binding, case, identity):
        self._active()
        if self.registry is not None and binding.registry is not self.registry:
            raise ValueError("Task binding belongs to another registry identity")
        if binding.profile != self.profile:
            raise ValueError("Task binding belongs to another runtime profile")
        return self.driver.execute(self.runtime, binding, case, identity)

    def observe(self, reset=False):
        self._active()
        return self.collector.snapshot(self.runtime, self.profile, reset=reset)

    def _active(self):
        if self.runtime is None or self.closed:
            raise RuntimeError("Driver session is not active")

    def close(self):
        if not self.closed:
            if self.runtime is not None:
                self.runtime.close()
            self.closed = True

    def __enter__(self):
        return self.start()

    def __exit__(self, *error):
        self.close()
