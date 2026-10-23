"""Single-host durable reference ExposureGate and independent trace oracle."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast


class GateState(str, Enum):
    PREDECESSOR_OPEN = "PREDECESSOR_OPEN"
    CANDIDATE_STAGED = "CANDIDATE_STAGED"
    CANDIDATE_CLOSED = "CANDIDATE_CLOSED"
    OPENING = "OPENING"
    CANDIDATE_OPEN = "CANDIDATE_OPEN"
    DRAINING = "DRAINING"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED_SAFE = "FAILED_SAFE"
    SAFETY_UNKNOWN = "SAFETY_UNKNOWN"


class RollbackClass(str, Enum):
    RESTORED_STRONG = "RESTORED_STRONG"
    BEHAVIORAL = "BEHAVIORAL"
    FAILED_SAFE = "FAILED_SAFE"
    SAFETY_UNKNOWN = "SAFETY_UNKNOWN"


class GateError(RuntimeError):
    pass


class InjectedCrash(GateError):
    pass


@dataclass(frozen=True)
class AdmissionRequest:
    request_id: str
    admitted_at: int


@dataclass(frozen=True)
class AdmissionResult:
    finished_at: int
    result: str | None = None
    abort: str | None = None


@dataclass(frozen=True)
class RouteState:
    generation: int | None
    fence: str
    snapshot: Any | None


@dataclass(frozen=True)
class LeaseGrant:
    holder: str
    fencing_token: int
    expires_at: int


class LeaseAuthority:
    def __init__(self, grant: LeaseGrant):
        self.grant = grant

    def current(self) -> LeaseGrant:
        return self.grant


class TrafficAuthorityAdapter(Protocol):
    """Route authority queried by the gate; no in-process route field required."""

    authority: LeaseAuthority
    behavioral_restore: bool

    def close(self, generation: int) -> None: ...

    def open(
        self,
        generation: int,
        fence: str,
        snapshot: dict[str, Any],
        grant: LeaseGrant,
        now: int,
    ) -> None: ...

    def close_all(self) -> None: ...

    def restore(
        self, generation: int, fence: str, snapshot: dict[str, Any]
    ) -> RouteState: ...

    def lease_valid(self, grant: LeaseGrant, now: int) -> bool: ...

    def query_route(self) -> RouteState: ...


@dataclass(frozen=True)
class OpenProof:
    plan_id: str
    generation: int
    evidence_count: int
    evidence_digest: str
    coverage_digest: str
    lease: LeaseGrant
    candidate_digest: str

    @property
    def digest(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BehavioralOracleResult:
    equivalent: bool
    observations_digest: str
    oracle: str


@dataclass(frozen=True)
class FailCloseResult:
    classification: RollbackClass
    actual_route: RouteState | None
    error: str | None


class DeterministicTrafficAdapter:
    """In-process scheduler admission model with deterministic fault points."""

    def __init__(
        self,
        predecessor_generation: int,
        predecessor_snapshot: dict[str, Any],
        *,
        authority: LeaseAuthority | None = None,
        faults: set[str] | None = None,
        behavioral_restore: bool = False,
        rollback_fails: bool = False,
    ):
        self.route = RouteState(
            predecessor_generation,
            f"fence-{predecessor_generation}",
            predecessor_snapshot,
        )
        self.authority = authority or LeaseAuthority(LeaseGrant("manager", 1, 2**62))
        self.faults = set() if faults is None else faults
        self.behavioral_restore = behavioral_restore
        self.rollback_fails = rollback_fails

    def _hit(self, point: str) -> None:
        if point in self.faults:
            self.faults.remove(point)
            raise InjectedCrash(point)

    def close(self, generation: int) -> None:
        self._hit("close")

    def open(
        self,
        generation: int,
        fence: str,
        snapshot: dict[str, Any],
        grant: LeaseGrant,
        now: int,
    ) -> None:
        self._hit("open.before")
        if self.authority.current() != grant or grant.expires_at < now:
            raise GateError("lease changed or expired before traffic open")
        self.route = RouteState(generation, fence, snapshot)
        self._hit("open.after")

    def close_all(self) -> None:
        self._hit("close_all")
        self.route = RouteState(None, "closed", None)

    def restore(
        self, generation: int, fence: str, snapshot: dict[str, Any]
    ) -> RouteState:
        if self.rollback_fails:
            raise GateError("rollback failed")
        restored = dict(snapshot)
        if self.behavioral_restore:
            restored = {"behaviorally_equivalent": True}
        self.route = RouteState(generation, fence, restored)
        return self.route

    def lease_valid(self, grant: LeaseGrant, now: int) -> bool:
        return self.authority.current() == grant and grant.expires_at >= now

    def query_route(self) -> RouteState:
        self._hit("route.query")
        return self.route


class ReferenceExposureGate:
    """Durable reference gate; not a proxy or Kubernetes readiness check."""

    exposure_gate_capability = "ecpa.reference-exposure-gate/0.2"
    schema_version = 2
    _SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")

    _ALLOWED = {
        GateState.PREDECESSOR_OPEN: {
            GateState.CANDIDATE_STAGED,
            GateState.FAILED_SAFE,
            GateState.SAFETY_UNKNOWN,
        },
        GateState.CANDIDATE_STAGED: {
            GateState.CANDIDATE_CLOSED,
            GateState.ROLLED_BACK,
            GateState.FAILED_SAFE,
            GateState.SAFETY_UNKNOWN,
        },
        GateState.CANDIDATE_CLOSED: {
            GateState.OPENING,
            GateState.ROLLED_BACK,
            GateState.FAILED_SAFE,
            GateState.SAFETY_UNKNOWN,
        },
        GateState.OPENING: {
            GateState.CANDIDATE_OPEN,
            GateState.FAILED_SAFE,
            GateState.SAFETY_UNKNOWN,
        },
        GateState.CANDIDATE_OPEN: {
            GateState.DRAINING,
            GateState.ROLLED_BACK,
            GateState.FAILED_SAFE,
            GateState.SAFETY_UNKNOWN,
        },
        GateState.DRAINING: {
            GateState.CANDIDATE_OPEN,
            GateState.ROLLED_BACK,
            GateState.FAILED_SAFE,
            GateState.SAFETY_UNKNOWN,
        },
        GateState.ROLLED_BACK: set(),
        GateState.FAILED_SAFE: set(),
        GateState.SAFETY_UNKNOWN: set(),
    }

    def __init__(
        self,
        path: str | Path,
        traffic: TrafficAuthorityAdapter,
        *,
        clock: Callable[[], int],
        authority: LeaseAuthority | None = None,
        faults: set[str] | None = None,
    ):
        # Every mutating context starts with BEGIN IMMEDIATE. Together with the
        # revision CAS this makes the documented single-writer contract explicit.
        self.connection = sqlite3.connect(path, isolation_level="IMMEDIATE")
        self.connection.row_factory = sqlite3.Row
        self.traffic = traffic
        self.clock = clock
        self.authority = authority or traffic.authority
        self.faults = set() if faults is None else faults
        self._schema()

    def _schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS gate_state(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1), state TEXT NOT NULL,
              predecessor_generation INTEGER NOT NULL, candidate_generation INTEGER,
              predecessor_json BLOB NOT NULL, candidate_json BLOB,
              route_generation INTEGER, route_fence TEXT NOT NULL,
              lease_json BLOB NOT NULL, transition_seq INTEGER NOT NULL,
              revision INTEGER NOT NULL, schema_version INTEGER NOT NULL,
              proof_digest TEXT, candidate_digest TEXT NOT NULL,
              observed_candidate_digest TEXT,
              predecessor_digest TEXT NOT NULL,
              observed_predecessor_digest TEXT);
            CREATE TABLE IF NOT EXISTS gate_transition(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, generation INTEGER NOT NULL,
              kind TEXT NOT NULL, operation TEXT NOT NULL,
              operation_id TEXT NOT NULL, from_state TEXT,
              to_state TEXT, fence TEXT, at INTEGER NOT NULL,
              detail_json BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS admission(
              request_id TEXT PRIMARY KEY, route_epoch TEXT NOT NULL,
              chosen_generation INTEGER NOT NULL, admitted_at INTEGER NOT NULL,
              finished_at INTEGER, result TEXT, abort TEXT,
              transition_seq INTEGER NOT NULL,
              CHECK((finished_at IS NULL AND result IS NULL AND abort IS NULL) OR
                    (finished_at >= admitted_at AND
                     ((result IS NOT NULL) != (abort IS NOT NULL)))));
            CREATE TABLE IF NOT EXISTS admission_event(
              seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
              kind TEXT NOT NULL, chosen_generation INTEGER NOT NULL,
              route_epoch TEXT NOT NULL, at INTEGER NOT NULL,
              transition_seq INTEGER NOT NULL, detail_json BLOB NOT NULL);
            """
        )
        self.connection.commit()

    @staticmethod
    def _json(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def _digest(cls, value: Any) -> str:
        return "sha256:" + hashlib.sha256(cls._json(value)).hexdigest()

    def _row(self) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM gate_state WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise GateError("gate is not initialized")
        if row["schema_version"] != self.schema_version:
            raise GateError("unknown gate schema version")
        return cast(sqlite3.Row, row)

    def _hit(self, point: str) -> None:
        if point in self.faults:
            self.faults.remove(point)
            raise InjectedCrash(point)

    def _query_route_or_none(self) -> RouteState | None:
        try:
            return self.traffic.query_route()
        except Exception:
            return None

    def _append(
        self,
        db: sqlite3.Connection,
        generation: int,
        kind: str,
        operation: str,
        from_state: GateState | None,
        to_state: GateState | None,
        fence: str | None,
        detail: dict[str, Any],
        operation_id: str | None = None,
    ) -> int:
        correlation = operation_id or f"{operation.split('.')[0]}:{generation}"
        cursor = db.execute(
            "INSERT INTO gate_transition(generation,kind,operation,operation_id,"
            "from_state,to_state,fence,at,detail_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                generation,
                kind,
                operation,
                correlation,
                None if from_state is None else from_state.value,
                None if to_state is None else to_state.value,
                fence,
                self.clock(),
                self._json(detail),
            ),
        )
        if cursor.lastrowid is None:
            raise GateError("gate transition insert did not produce a row ID")
        return cursor.lastrowid

    def _transition(
        self,
        db: sqlite3.Connection,
        generation: int,
        target: GateState,
        operation: str,
        kind: str,
        detail: dict[str, Any],
        fence: str | None = None,
        operation_id: str | None = None,
    ) -> int:
        row = self._row()
        current = GateState(row["state"])
        if target != current and target not in self._ALLOWED[current]:
            raise GateError(f"invalid gate transition {current.value}->{target.value}")
        seq = self._append(
            db,
            generation,
            kind,
            operation,
            current,
            target,
            fence,
            detail,
            operation_id,
        )
        changed = db.execute(
            "UPDATE gate_state SET state=?,transition_seq=?,revision=revision+1 "
            "WHERE singleton=1 AND revision=?",
            (target.value, seq, row["revision"]),
        ).rowcount
        if changed != 1:
            raise GateError("gate revision CAS mismatch")
        return seq

    def stage(self, candidate: Any, predecessor: Any) -> int:
        predecessor_value = (
            asdict(predecessor)
            if hasattr(predecessor, "__dataclass_fields__")
            else predecessor
        )
        candidate_value = (
            asdict(candidate)
            if hasattr(candidate, "__dataclass_fields__")
            else candidate
        )
        predecessor_generation = int(predecessor_value["generation"])
        candidate_generation = predecessor_generation + 1
        candidate_digest = self._digest(candidate_value)
        predecessor_digest = self._digest(predecessor_value)
        existing = self.connection.execute("SELECT * FROM gate_state").fetchone()
        if existing is not None:
            if (
                existing["candidate_generation"] == candidate_generation
                and json.loads(existing["candidate_json"]) == candidate_value
                and json.loads(existing["predecessor_json"]) == predecessor_value
                and existing["route_fence"] == f"fence-{predecessor_generation}"
            ):
                return candidate_generation
            raise GateError("another candidate is already staged")
        route = self.traffic.query_route()
        if (
            route.generation != predecessor_generation
            or route.fence != f"fence-{predecessor_generation}"
            or self._json(route.snapshot) != self._json(predecessor_value)
        ):
            raise GateError("actual predecessor generation/fence/snapshot mismatch")
        with self.connection as db:
            db.execute(
                "INSERT INTO gate_state VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    GateState.PREDECESSOR_OPEN.value,
                    predecessor_generation,
                    candidate_generation,
                    self._json(predecessor_value),
                    self._json(candidate_value),
                    route.generation,
                    route.fence,
                    self._json(asdict(self.authority.current())),
                    0,
                    0,
                    self.schema_version,
                    None,
                    candidate_digest,
                    None,
                    predecessor_digest,
                    predecessor_digest,
                ),
            )
            self._transition(
                db,
                candidate_generation,
                GateState.CANDIDATE_STAGED,
                "stage",
                "receipt",
                {
                    "candidate": candidate_value,
                    "candidate_digest": candidate_digest,
                    "predecessor_digest": predecessor_digest,
                },
            )
        return candidate_generation

    def close(self, generation: int) -> None:
        row = self._row()
        state = GateState(row["state"])
        if row["candidate_generation"] != generation:
            raise GateError("stale generation close")
        if state is GateState.CANDIDATE_CLOSED:
            return
        operation_id = f"close:{generation}"
        pending = self.connection.execute(
            "SELECT 1 FROM gate_transition WHERE operation_id=? AND kind='intent' "
            "AND NOT EXISTS (SELECT 1 FROM gate_transition r "
            "WHERE r.operation_id=? AND r.kind='receipt')",
            (operation_id, operation_id),
        ).fetchone()
        if pending is None:
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "intent",
                    "close",
                    state,
                    None,
                    None,
                    {},
                    operation_id,
                )
        self._hit("close.before_effect")
        self.traffic.close(generation)
        self._hit("close.after_effect")
        actual = self.traffic.query_route()
        self._hit("close.after_query")
        observed_digest = self._digest(actual.snapshot)
        if (
            actual.generation != row["predecessor_generation"]
            or actual.fence != row["route_fence"]
            or observed_digest != row["predecessor_digest"]
        ):
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "receipt",
                    "close",
                    state,
                    None,
                    actual.fence,
                    {
                        "cancelled": "predecessor continuity mismatch",
                        "observed_predecessor_digest": observed_digest,
                    },
                    operation_id=operation_id,
                )
            self.fail_close(generation, "predecessor continuity mismatch")
            raise GateError("close changed predecessor route identity")
        with self.connection as db:
            self._transition(
                db,
                generation,
                GateState.CANDIDATE_CLOSED,
                "close",
                "receipt",
                {
                    "actual_route": asdict(actual),
                    "observed_predecessor_digest": observed_digest,
                },
                operation_id=operation_id,
            )
            db.execute(
                "UPDATE gate_state SET observed_predecessor_digest=? WHERE singleton=1",
                (observed_digest,),
            )

    def open(self, generation: int, proof: OpenProof) -> str:
        row = self._row()
        if proof.generation != generation or row["candidate_generation"] != generation:
            raise GateError("stale generation open")
        candidate = json.loads(row["candidate_json"])
        expected_plan = (
            candidate if isinstance(candidate, str) else candidate["plan_id"]
        )
        if proof.plan_id != expected_plan:
            raise GateError("open proof plan mismatch")
        if proof.candidate_digest != row["candidate_digest"]:
            raise GateError("open proof candidate snapshot mismatch")
        if (
            proof.evidence_count < 1
            or self._SHA256.fullmatch(proof.evidence_digest) is None
            or self._SHA256.fullmatch(proof.coverage_digest) is None
        ):
            raise GateError("open proof has no durable evidence coverage")
        staged_lease = LeaseGrant(**json.loads(row["lease_json"]))
        if proof.lease != staged_lease or not self.traffic.lease_valid(
            proof.lease, self.clock()
        ):
            raise GateError("lease lost")
        state = GateState(row["state"])
        if state is GateState.CANDIDATE_OPEN:
            if row["proof_digest"] != proof.digest:
                raise GateError("idempotent open proof mismatch")
            return str(row["route_fence"])
        if state is GateState.OPENING:
            raise GateError("opening outcome requires reconcile")
        fence = (
            f"gate-{generation}-lease-{proof.lease.fencing_token}-"
            f"{proof.candidate_digest[7:19]}-{row['transition_seq'] + 1}"
        )
        self._hit("open.before_intent")
        with self.connection as db:
            self._transition(
                db,
                generation,
                GateState.OPENING,
                "open",
                "intent",
                {**asdict(proof), "proof_digest": proof.digest},
                fence,
            )
            db.execute(
                "UPDATE gate_state SET proof_digest=? WHERE singleton=1",
                (proof.digest,),
            )
        self._hit("open.after_intent")
        self.traffic.open(generation, fence, candidate, proof.lease, self.clock())
        self._hit("open.after_side_effect")
        actual = self.traffic.query_route()
        self._hit("open.after_query")
        actual_digest = self._digest(actual.snapshot)
        if (
            actual.generation != generation
            or actual.fence != fence
            or actual_digest != proof.candidate_digest
        ):
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "receipt",
                    "open",
                    GateState.OPENING,
                    None,
                    fence,
                    {"cancelled": "actual candidate route mismatch"},
                    operation_id=f"open:{generation}",
                )
            self.fail_close(generation, "actual candidate route mismatch")
            raise GateError("actual candidate route mismatch")
        with self.connection as db:
            seq = self._transition(
                db,
                generation,
                GateState.CANDIDATE_OPEN,
                "open",
                "receipt",
                {
                    "actual_route": generation,
                    "lease_fencing_token": proof.lease.fencing_token,
                    "proof_digest": proof.digest,
                    "observed_candidate_digest": actual_digest,
                },
                fence,
            )
            db.execute(
                "UPDATE gate_state SET route_generation=?,route_fence=?,"
                "transition_seq=?,observed_candidate_digest=? WHERE singleton=1",
                (generation, fence, seq, actual_digest),
            )
        return fence

    def drain(self, generation: int) -> bool:
        row = self._row()
        if row["candidate_generation"] != generation:
            raise GateError("stale generation drain")
        state = GateState(row["state"])
        if state is not GateState.DRAINING:
            with self.connection as db:
                self._transition(
                    db, generation, GateState.DRAINING, "drain", "intent", {}
                )
        unfinished = self.connection.execute(
            "SELECT COUNT(*) FROM admission WHERE chosen_generation=? "
            "AND finished_at IS NULL",
            (row["predecessor_generation"],),
        ).fetchone()[0]
        if unfinished:
            return False
        with self.connection as db:
            self._transition(
                db,
                generation,
                GateState.CANDIDATE_OPEN,
                "drain",
                "receipt",
                {"unfinished_predecessor": 0},
            )
        return True

    def observe(
        self, request: AdmissionRequest, result: AdmissionResult | None = None
    ) -> int:
        existing = self.connection.execute(
            "SELECT * FROM admission WHERE request_id=?", (request.request_id,)
        ).fetchone()
        if existing is None:
            gate_state = GateState(self._row()["state"])
            if gate_state in {
                GateState.OPENING,
                GateState.ROLLED_BACK,
                GateState.FAILED_SAFE,
                GateState.SAFETY_UNKNOWN,
            }:
                raise GateError(f"admission blocked in {gate_state.value}")
            actual = self.traffic.query_route()
            row = self._row()
            if actual.generation is None or actual.fence == "closed":
                raise GateError("traffic is failed safe")
            predecessor_mismatch = gate_state in {
                GateState.PREDECESSOR_OPEN,
                GateState.CANDIDATE_STAGED,
                GateState.CANDIDATE_CLOSED,
            } and (
                actual.generation != row["predecessor_generation"]
                or actual.fence != f"fence-{row['predecessor_generation']}"
                or self._digest(actual.snapshot) != row["predecessor_digest"]
            )
            if predecessor_mismatch:
                self.fail_close(
                    int(row["candidate_generation"]),
                    "pre-open predecessor continuity mismatch",
                )
                raise GateError("predecessor route identity changed")
            if (
                actual.generation != row["route_generation"]
                or actual.fence != row["route_fence"]
                or (
                    actual.generation == row["candidate_generation"]
                    and self._digest(actual.snapshot)
                    != row["observed_candidate_digest"]
                )
            ):
                raise GateError("unknown route state")
            with self.connection as db:
                db.execute(
                    "INSERT INTO admission VALUES(?,?,?,?,?,?,?,?)",
                    (
                        request.request_id,
                        actual.fence,
                        actual.generation,
                        request.admitted_at,
                        None,
                        None,
                        None,
                        row["transition_seq"],
                    ),
                )
                db.execute(
                    "INSERT INTO admission_event(request_id,kind,chosen_generation,"
                    "route_epoch,at,transition_seq,detail_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        request.request_id,
                        "admit",
                        actual.generation,
                        actual.fence,
                        request.admitted_at,
                        row["transition_seq"],
                        self._json({}),
                    ),
                )
            existing = self.connection.execute(
                "SELECT * FROM admission WHERE request_id=?", (request.request_id,)
            ).fetchone()
        if result is not None:
            if (result.result is None) == (result.abort is None):
                raise GateError("finish requires exactly one of result or abort")
            if result.finished_at < existing["admitted_at"]:
                raise GateError("finish precedes admission")
            if existing["finished_at"] is not None:
                raise GateError("request finish is immutable")
            else:
                with self.connection as db:
                    changed = db.execute(
                        "UPDATE admission SET finished_at=?,result=?,abort=? "
                        "WHERE request_id=? AND finished_at IS NULL",
                        (
                            result.finished_at,
                            result.result,
                            result.abort,
                            request.request_id,
                        ),
                    ).rowcount
                    if changed != 1:
                        raise GateError("concurrent request finish")
                    db.execute(
                        "INSERT INTO admission_event(request_id,kind,"
                        "chosen_generation,route_epoch,at,transition_seq,detail_json) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (
                            request.request_id,
                            "finish",
                            existing["chosen_generation"],
                            existing["route_epoch"],
                            result.finished_at,
                            existing["transition_seq"],
                            self._json(
                                {"result": result.result, "abort": result.abort}
                            ),
                        ),
                    )
        return int(existing["chosen_generation"])

    def reconcile(self) -> GateState:
        row = self._row()
        state = GateState(row["state"])
        fail_close_intent = self.connection.execute(
            "SELECT * FROM gate_transition WHERE operation='fail_close' "
            "AND kind='intent' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        fail_close_receipt = self.connection.execute(
            "SELECT * FROM gate_transition WHERE operation='fail_close' "
            "AND kind='receipt' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if fail_close_intent is not None and (
            fail_close_receipt is None
            or fail_close_receipt["seq"] < fail_close_intent["seq"]
        ):
            try:
                actual = self.traffic.query_route()
                error = None
            except Exception as exc:
                actual = None
                error = str(exc)
            closed = (
                actual is not None
                and actual.generation is None
                and actual.fence == "closed"
            )
            target = GateState.FAILED_SAFE if closed else GateState.SAFETY_UNKNOWN
            classification = (
                RollbackClass.FAILED_SAFE if closed else RollbackClass.SAFETY_UNKNOWN
            )
            with self.connection as db:
                self._transition(
                    db,
                    int(row["candidate_generation"]),
                    target,
                    "fail_close",
                    "receipt",
                    {
                        "classification": classification.value,
                        "reconciled": True,
                        "actual_route": None if actual is None else asdict(actual),
                        "error": error,
                    },
                    operation_id=fail_close_intent["operation_id"],
                )
                db.execute(
                    "UPDATE gate_state SET route_generation=?,route_fence=? "
                    "WHERE singleton=1",
                    (
                        None if actual is None else actual.generation,
                        "unknown" if actual is None else actual.fence,
                    ),
                )
            return target
        recovery_close_intent = self.connection.execute(
            "SELECT * FROM gate_transition WHERE operation='open.reconcile.close_all' "
            "AND kind='intent' AND NOT EXISTS (SELECT 1 FROM gate_transition r "
            "WHERE r.operation_id=gate_transition.operation_id AND r.kind='receipt') "
            "ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if recovery_close_intent is not None:
            generation = int(row["candidate_generation"])
            self._hit("open_recovery_close.before_effect")
            with suppress(Exception):
                self.traffic.close_all()
            self._hit("open_recovery_close.after_effect")
            try:
                actual = self.traffic.query_route()
            except Exception:
                actual = None
            self._hit("open_recovery_close.after_query")
            closed = (
                actual is not None
                and actual.generation is None
                and actual.fence == "closed"
            )
            target = GateState.FAILED_SAFE if closed else GateState.SAFETY_UNKNOWN
            open_intent = self.connection.execute(
                "SELECT * FROM gate_transition WHERE operation='open' "
                "AND kind='intent' ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "receipt",
                    "open.reconcile.close_all",
                    GateState.OPENING,
                    None,
                    "closed",
                    {"actual_route": None if actual is None else asdict(actual)},
                    operation_id=recovery_close_intent["operation_id"],
                )
                self._transition(
                    db,
                    generation,
                    target,
                    "open",
                    "receipt",
                    {"cancelled": "recovery close", "closed": closed},
                    operation_id=open_intent["operation_id"],
                )
            return target
        rollback_intent = self.connection.execute(
            "SELECT * FROM gate_transition WHERE operation='rollback' "
            "AND kind='intent' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        rollback_receipt = self.connection.execute(
            "SELECT * FROM gate_transition WHERE operation='rollback' "
            "AND kind='receipt' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if rollback_intent is not None and (
            rollback_receipt is None or rollback_receipt["seq"] < rollback_intent["seq"]
        ):
            predecessor = json.loads(row["predecessor_json"])
            expected_fence = f"fence-{row['predecessor_generation']}"
            try:
                actual = self.traffic.query_route()
            except Exception:
                actual = None
            if (
                actual is not None
                and actual.generation == row["predecessor_generation"]
                and actual.fence == expected_fence
                and self._json(actual.snapshot) == self._json(predecessor)
            ):
                target = GateState.ROLLED_BACK
                classification = RollbackClass.RESTORED_STRONG
            elif (
                actual is not None
                and actual.generation is None
                and actual.fence == "closed"
            ):
                target = GateState.FAILED_SAFE
                classification = RollbackClass.FAILED_SAFE
            else:
                target = GateState.SAFETY_UNKNOWN
                classification = RollbackClass.SAFETY_UNKNOWN
            with self.connection as db:
                self._transition(
                    db,
                    int(row["candidate_generation"]),
                    target,
                    "rollback",
                    "receipt",
                    {
                        "classification": classification.value,
                        "reconciled": True,
                        "actual_route": None if actual is None else asdict(actual),
                    },
                    operation_id=rollback_intent["operation_id"],
                )
            return target
        if state is not GateState.OPENING:
            return state
        generation = int(row["candidate_generation"])
        intent = self.connection.execute(
            "SELECT * FROM gate_transition WHERE operation='open' AND kind='intent' "
            "ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        detail = json.loads(intent["detail_json"])
        proof = OpenProof(
            detail["plan_id"],
            detail["generation"],
            detail["evidence_count"],
            detail["evidence_digest"],
            detail["coverage_digest"],
            LeaseGrant(**detail["lease"]),
            detail["candidate_digest"],
        )
        if (
            proof.digest != detail["proof_digest"]
            or proof.lease != LeaseGrant(**json.loads(row["lease_json"]))
            or not self.traffic.lease_valid(proof.lease, self.clock())
        ):
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "receipt",
                    "open",
                    GateState.OPENING,
                    None,
                    intent["fence"],
                    {"cancelled": "persisted open lease is stale"},
                    operation_id=intent["operation_id"],
                )
            outcome = self.fail_close(generation, "persisted open lease is stale")
            return (
                GateState.FAILED_SAFE
                if outcome.classification is RollbackClass.FAILED_SAFE
                else GateState.SAFETY_UNKNOWN
            )
        try:
            actual = self.traffic.query_route()
        except Exception:
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "receipt",
                    "open",
                    GateState.OPENING,
                    None,
                    intent["fence"],
                    {"cancelled": "open recovery route query failed"},
                    operation_id=intent["operation_id"],
                )
            outcome = self.fail_close(generation, "open recovery route query failed")
            return (
                GateState.FAILED_SAFE
                if outcome.classification is RollbackClass.FAILED_SAFE
                else GateState.SAFETY_UNKNOWN
            )
        actual_digest = self._digest(actual.snapshot)
        if (
            actual.generation == generation
            and actual.fence == intent["fence"]
            and actual_digest == proof.candidate_digest
        ):
            with self.connection as db:
                seq = self._transition(
                    db,
                    generation,
                    GateState.CANDIDATE_OPEN,
                    "open.reconcile",
                    "receipt",
                    {
                        "queried_actual_route": generation,
                        "observed_candidate_digest": actual_digest,
                    },
                    actual.fence,
                    intent["operation_id"],
                )
                db.execute(
                    "UPDATE gate_state SET route_generation=?,route_fence=?,"
                    "transition_seq=?,observed_candidate_digest=? WHERE singleton=1",
                    (generation, actual.fence, seq, actual_digest),
                )
            return GateState.CANDIDATE_OPEN
        with self.connection as db:
            self._append(
                db,
                generation,
                "intent",
                "open.reconcile.close_all",
                GateState.OPENING,
                None,
                "closed",
                {"queried_actual_generation": actual.generation},
                operation_id=f"open-reconcile-close:{generation}",
            )
        close_error = None
        self._hit("open_recovery_close.before_effect")
        try:
            self.traffic.close_all()
        except Exception as exc:
            close_error = str(exc)
        self._hit("open_recovery_close.after_effect")
        try:
            closed_route = self.traffic.query_route()
        except Exception as exc:
            closed_route = None
            close_error = f"{close_error}; {exc}" if close_error else str(exc)
        self._hit("open_recovery_close.after_query")
        with self.connection as db:
            self._append(
                db,
                generation,
                "receipt",
                "open.reconcile.close_all",
                GateState.OPENING,
                None,
                "closed",
                {
                    "actual_generation": actual.generation,
                    "actual_route": (
                        None if closed_route is None else asdict(closed_route)
                    ),
                    "error": close_error,
                },
                operation_id=f"open-reconcile-close:{generation}",
            )
        if close_error is not None:
            with self.connection as db:
                self._append(
                    db,
                    generation,
                    "receipt",
                    "open",
                    GateState.OPENING,
                    None,
                    intent["fence"],
                    {"cancelled": "open recovery close failed"},
                    operation_id=intent["operation_id"],
                )
            outcome = self.fail_close(generation, "open recovery close failed")
            return (
                GateState.FAILED_SAFE
                if outcome.classification is RollbackClass.FAILED_SAFE
                else GateState.SAFETY_UNKNOWN
            )
        with self.connection as db:
            self._transition(
                db,
                generation,
                GateState.FAILED_SAFE,
                "open",
                "receipt",
                {"actual_generation": actual.generation, "action": "close_all"},
                operation_id=intent["operation_id"],
            )
            db.execute(
                "UPDATE gate_state SET route_generation=NULL,route_fence='closed' "
                "WHERE singleton=1"
            )
        return GateState.FAILED_SAFE

    def rollback(
        self,
        generation: int,
        behavioral_oracle: BehavioralOracleResult | None = None,
    ) -> RollbackClass:
        row = self._row()
        if row["candidate_generation"] != generation:
            raise GateError("stale generation rollback")
        predecessor = json.loads(row["predecessor_json"])
        predecessor_generation = int(row["predecessor_generation"])
        predecessor_fence = f"fence-{predecessor_generation}"
        with self.connection as db:
            self._append(
                db,
                generation,
                "intent",
                "rollback",
                GateState(row["state"]),
                None,
                predecessor_fence,
                {
                    "behavioral_oracle": (
                        None if behavioral_oracle is None else asdict(behavioral_oracle)
                    )
                },
            )
        self._hit("rollback.after_intent")
        try:
            self.traffic.restore(predecessor_generation, predecessor_fence, predecessor)
        except Exception:
            with suppress(Exception):
                self.traffic.close_all()
            actual = self._query_route_or_none()
            outcome = (
                RollbackClass.FAILED_SAFE
                if actual is not None
                and actual.generation is None
                and actual.fence == "closed"
                else RollbackClass.SAFETY_UNKNOWN
            )
        else:
            actual = self._query_route_or_none()
            if (
                actual is not None
                and actual.generation == predecessor_generation
                and actual.snapshot == predecessor
                and actual.fence == predecessor_fence
            ):
                outcome = RollbackClass.RESTORED_STRONG
            elif (
                behavioral_oracle is not None
                and behavioral_oracle.equivalent
                and behavioral_oracle.observations_digest.startswith("sha256:")
                and behavioral_oracle.oracle
                and self.traffic.behavioral_restore
            ):
                outcome = RollbackClass.BEHAVIORAL
            else:
                with suppress(Exception):
                    self.traffic.close_all()
                actual = self._query_route_or_none()
                outcome = (
                    RollbackClass.FAILED_SAFE
                    if actual is not None
                    and actual.generation is None
                    and actual.fence == "closed"
                    else RollbackClass.SAFETY_UNKNOWN
                )
        self._hit("rollback.after_effect")
        target = {
            RollbackClass.RESTORED_STRONG: GateState.ROLLED_BACK,
            RollbackClass.BEHAVIORAL: GateState.ROLLED_BACK,
            RollbackClass.FAILED_SAFE: GateState.FAILED_SAFE,
            RollbackClass.SAFETY_UNKNOWN: GateState.SAFETY_UNKNOWN,
        }[outcome]
        with self.connection as db:
            self._transition(
                db,
                generation,
                target,
                "rollback",
                "receipt",
                {
                    "classification": outcome.value,
                    "behavioral_oracle": (
                        None if behavioral_oracle is None else asdict(behavioral_oracle)
                    ),
                    "actual_route": None if actual is None else asdict(actual),
                },
            )
            db.execute(
                "UPDATE gate_state SET route_generation=?,route_fence=? "
                "WHERE singleton=1",
                (
                    None if actual is None else actual.generation,
                    "unknown" if actual is None else actual.fence,
                ),
            )
        return outcome

    def fail_close(self, generation: int, reason: str) -> FailCloseResult:
        row = self._row()
        if row["candidate_generation"] != generation:
            raise GateError("stale generation fail-close")
        state = GateState(row["state"])
        with self.connection as db:
            self._append(
                db,
                generation,
                "intent",
                "fail_close",
                state,
                None,
                "closed",
                {"reason": reason},
            )
        self._hit("fail_close.before_effect")
        error = None
        try:
            self.traffic.close_all()
        except Exception as exc:
            error = str(exc)
        self._hit("fail_close.after_effect")
        try:
            actual = self.traffic.query_route()
        except Exception as exc:
            actual = None
            error = f"{error}; route query: {exc}" if error else str(exc)
        closed = (
            actual is not None
            and actual.generation is None
            and actual.fence == "closed"
        )
        classification = (
            RollbackClass.FAILED_SAFE if closed else RollbackClass.SAFETY_UNKNOWN
        )
        target = GateState.FAILED_SAFE if closed else GateState.SAFETY_UNKNOWN
        with self.connection as db:
            self._transition(
                db,
                generation,
                target,
                "fail_close",
                "receipt",
                {
                    "classification": classification.value,
                    "actual_route": None if actual is None else asdict(actual),
                    "error": error,
                },
            )
            db.execute(
                "UPDATE gate_state SET route_generation=?,route_fence=? "
                "WHERE singleton=1",
                (
                    None if actual is None else actual.generation,
                    "unknown" if actual is None else actual.fence,
                ),
            )
        return FailCloseResult(classification, actual, error)

    def close_store(self) -> None:
        self.connection.close()


def evaluate_trace(path: str | Path) -> dict[str, Any]:
    """Offline oracle over raw rows; does not call gate decision methods."""
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    transitions = db.execute("SELECT * FROM gate_transition ORDER BY seq").fetchall()
    admissions = db.execute("SELECT * FROM admission ORDER BY admitted_at").fetchall()
    admission_events = db.execute(
        "SELECT * FROM admission_event ORDER BY seq"
    ).fetchall()
    state = db.execute("SELECT * FROM gate_state WHERE singleton=1").fetchone()
    errors: list[str] = []
    open_receipts = [
        row
        for row in transitions
        if row["kind"] == "receipt"
        and row["to_state"] == GateState.CANDIDATE_OPEN.value
        and row["operation"].startswith("open")
    ]
    candidate = None if state is None else state["candidate_generation"]
    predecessor = None if state is None else state["predecessor_generation"]
    first_open_seq = None if not open_receipts else open_receipts[0]["seq"]
    open_fence = None if not open_receipts else open_receipts[0]["fence"]
    if state is None or not transitions:
        errors.append("missing gate state or transitions")
    expected_seq = 1
    replay_state = GateState.PREDECESSOR_OPEN.value
    allowed = {
        GateState.PREDECESSOR_OPEN.value: {
            GateState.CANDIDATE_STAGED.value,
            GateState.FAILED_SAFE.value,
            GateState.SAFETY_UNKNOWN.value,
        },
        GateState.CANDIDATE_STAGED.value: {
            GateState.CANDIDATE_CLOSED.value,
            GateState.ROLLED_BACK.value,
            GateState.FAILED_SAFE.value,
            GateState.SAFETY_UNKNOWN.value,
        },
        GateState.CANDIDATE_CLOSED.value: {
            GateState.OPENING.value,
            GateState.ROLLED_BACK.value,
            GateState.FAILED_SAFE.value,
            GateState.SAFETY_UNKNOWN.value,
        },
        GateState.OPENING.value: {
            GateState.CANDIDATE_OPEN.value,
            GateState.FAILED_SAFE.value,
            GateState.SAFETY_UNKNOWN.value,
        },
        GateState.CANDIDATE_OPEN.value: {
            GateState.DRAINING.value,
            GateState.ROLLED_BACK.value,
            GateState.FAILED_SAFE.value,
            GateState.SAFETY_UNKNOWN.value,
        },
        GateState.DRAINING.value: {GateState.CANDIDATE_OPEN.value},
    }
    pending: dict[tuple[int, str], int] = {}
    for row in transitions:
        if row["seq"] != expected_seq:
            errors.append(f"transition sequence gap at {expected_seq}")
            expected_seq = row["seq"]
        expected_seq += 1
        operation_id = row["operation_id"]
        if row["from_state"] != replay_state:
            errors.append(
                f"transition source mismatch at {row['seq']}: "
                f"{row['from_state']} != {replay_state}"
            )
        if row["kind"] == "intent":
            if operation_id in pending:
                errors.append(f"duplicate pending intent: {operation_id}")
            pending[operation_id] = row["seq"]
        elif row["kind"] == "receipt" and row["operation"] != "stage":
            if operation_id not in pending:
                errors.append(f"receipt without intent: {operation_id}")
            else:
                pending.pop(operation_id)
        target = row["to_state"]
        if target is not None and target != replay_state:
            if target not in allowed.get(replay_state, set()):
                errors.append(f"unknown transition: {replay_state}->{target}")
            replay_state = target
        if candidate is not None and row["generation"] != candidate:
            errors.append(f"transition generation mismatch at {row['seq']}")
    if pending:
        errors.append(f"unpaired intents: {sorted(pending)}")
    if state is not None and replay_state != state["state"]:
        errors.append("replayed final state does not match durable state")
    if len({row["generation"] for row in open_receipts}) > 1:
        errors.append("multiple open generations")
    if (
        state is not None
        and state["state"] == GateState.CANDIDATE_OPEN.value
        and (
            state["route_generation"] != candidate
            or state["route_fence"] != open_fence
            or state["observed_candidate_digest"] != state["candidate_digest"]
        )
    ):
        errors.append("final route observation does not match open receipt")
    for receipt in open_receipts:
        detail = json.loads(receipt["detail_json"])
        if detail.get("observed_candidate_digest") != state["candidate_digest"]:
            errors.append("open receipt candidate snapshot mismatch")
    close_receipts = [
        row
        for row in transitions
        if row["operation"] == "close" and row["kind"] == "receipt"
    ]
    for receipt in close_receipts:
        detail = json.loads(receipt["detail_json"])
        observed = detail.get("observed_predecessor_digest")
        if (
            "cancelled" not in detail
            and observed is not None
            and observed != state["predecessor_digest"]
        ):
            errors.append("close receipt predecessor snapshot mismatch")
    seen: dict[str, int] = {}
    for event in admission_events:
        prior = seen.setdefault(event["request_id"], event["chosen_generation"])
        if prior != event["chosen_generation"]:
            errors.append(f"request {event['request_id']} crossed generation")
    for row in admissions:
        if (
            candidate is not None
            and row["chosen_generation"] == candidate
            and (first_open_seq is None or row["transition_seq"] < first_open_seq)
        ):
            errors.append(f"candidate admitted before open: {row['request_id']}")
        if (
            predecessor is not None
            and row["chosen_generation"] == predecessor
            and first_open_seq is not None
            and row["transition_seq"] >= first_open_seq
        ):
            errors.append(f"predecessor admitted after open: {row['request_id']}")
        if row["chosen_generation"] not in {candidate, predecessor}:
            errors.append(f"unknown generation: {row['request_id']}")
        if not row["route_epoch"]:
            errors.append(f"missing route fence: {row['request_id']}")
        if row["chosen_generation"] == candidate and row["route_epoch"] != open_fence:
            errors.append(f"candidate fence mismatch: {row['request_id']}")
        if (
            row["chosen_generation"] == predecessor
            and row["route_epoch"] != f"fence-{predecessor}"
        ):
            errors.append(f"predecessor fence mismatch: {row['request_id']}")
    verdict = "PASS" if not errors else "FAIL"
    result = {
        "schema": "ecpa-exposure-oracle/0.1",
        "verdict": verdict,
        "errors": errors,
        "transition_count": len(transitions) if transitions else None,
        "admission_count": len(admissions) if admissions else None,
        "candidate_generation": candidate,
        "predecessor_generation": predecessor,
        "first_candidate_open_seq": first_open_seq,
    }
    db.close()
    return result


def export_trace(path: str | Path) -> list[dict[str, Any]]:
    """Export exact logical rows in deterministic JSON-serializable form."""
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    exported: list[dict[str, Any]] = []
    for row in db.execute("SELECT * FROM gate_transition ORDER BY seq"):
        item = dict(row)
        item["detail"] = json.loads(item.pop("detail_json"))
        item["classification"] = "synthetic/reference"
        exported.append(item)
    for row in db.execute("SELECT * FROM admission ORDER BY admitted_at"):
        item = dict(row)
        item["type"] = "admission"
        item["classification"] = "synthetic/reference"
        exported.append(item)
    for row in db.execute("SELECT * FROM admission_event ORDER BY seq"):
        item = dict(row)
        item["detail"] = json.loads(item.pop("detail_json"))
        item["type"] = "admission_event"
        item["classification"] = "synthetic/reference"
        exported.append(item)
    db.close()
    return exported
