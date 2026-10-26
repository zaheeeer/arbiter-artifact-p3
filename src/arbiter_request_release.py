"""Request-bound release protocol for the ARBITER closure campaign.

This module is intentionally separate from the audited-v4 implementation.  It
does not change the immutable 00--04 evidence.  It closes one specific API
defect: a token must authorize exactly one request, buyer/session, asset, quote,
query, and immutable answer buffer under one verified constraint snapshot.

The durable boundary is a local SQLite transaction.  A successful commit
inserts a purchase authorization and changes the staged answer to ``released``
in the same transaction.  Subsequent network delivery is recoverable from the
outbox but is outside this in-process prototype's atomicity claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import base64
import datetime as _datetime
from decimal import Decimal
import hashlib
import hmac
import json
import math
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

import arbiter_core as _core


PROTOCOL_VERSION = 2
PROTOCOL_PURPOSE = "arbiter.request_bound_release"
DEFAULT_TOKEN_TTL_NS = 5 * 60 * 1_000_000_000
MAX_ID_UTF8_BYTES = 512
MAX_SQL_UTF8_BYTES = 4 * 1024 * 1024


def _sha256_hex(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _validated_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty and have no edge whitespace")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_ID_UTF8_BYTES:
        raise ValueError(f"{name} exceeds {MAX_ID_UTF8_BYTES} UTF-8 bytes")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _validated_digest(name: str, value: Any) -> str:
    digest = _validated_identifier(name, value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return digest


def _validated_tolerance(value: Any) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("verifier tolerance must be finite and strictly positive")
    return tolerance


def _float64_hex(value: Any, *, nonnegative: bool = False) -> str:
    number = float(np.float64(value))
    if not math.isfinite(number):
        raise ValueError("numeric binding fields must be finite")
    if nonnegative and number < 0.0:
        raise ValueError("quoted price must be nonnegative")
    return number.hex()


def _same_float64(left: Any, right: Any) -> bool:
    return np.float64(left).tobytes() == np.float64(right).tobytes()


def exact_sql_digest(sql: str) -> str:
    """Digest exact SQL text; no unsound equivalence normalization is assumed."""

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a nonempty string")
    encoded = sql.encode("utf-8")
    if len(encoded) > MAX_SQL_UTF8_BYTES:
        raise ValueError(f"sql exceeds {MAX_SQL_UTF8_BYTES} UTF-8 bytes")
    return _sha256_hex(b"ARBITER-SQL-V1\x00", encoded)


def _answer_node(value: Any) -> Any:
    """Convert supported answer values to an unambiguous typed JSON tree."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "+inf" if value > 0 else "-inf"]
        return ["float", value.hex()]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, bytearray):
        return ["bytes", base64.b64encode(bytes(value)).decode("ascii")]
    if isinstance(value, _datetime.datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, _datetime.date):
        return ["date", value.isoformat()]
    if isinstance(value, _datetime.time):
        return ["time", value.isoformat()]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_answer_node(item) for item in value]]
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            key_node = _answer_node(key)
            item_node = _answer_node(item)
            items.append((_canonical_json(key_node), key_node, item_node))
        items.sort(key=lambda record: record[0])
        return ["mapping", [[key_node, item_node] for _, key_node, item_node in items]]
    raise TypeError(f"unsupported answer value type: {type(value).__name__}")


def canonical_answer_bytes(payload: Any) -> bytes:
    return _canonical_json(["ARBITER-ANSWER-V1", _answer_node(payload)])


def answer_digest(payload_bytes: bytes) -> str:
    if not isinstance(payload_bytes, bytes):
        raise TypeError("payload_bytes must be immutable bytes")
    return _sha256_hex(b"ARBITER-ANSWER-DIGEST-V1\x00", payload_bytes)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    buyer_id: str
    session_id: str
    asset_id: str
    query_digest: str
    quoted_price_hex: str
    currency: str = "EXPERIMENTAL_PRICE_UNIT"

    def __post_init__(self) -> None:
        for name in ["request_id", "buyer_id", "session_id", "asset_id", "currency"]:
            object.__setattr__(self, name, _validated_identifier(name, getattr(self, name)))
        object.__setattr__(self, "query_digest", _validated_digest("query_digest", self.query_digest))
        parsed_price = float.fromhex(_validated_identifier("quoted_price_hex", self.quoted_price_hex))
        if not math.isfinite(parsed_price) or parsed_price < 0.0:
            raise ValueError("quoted_price_hex must encode a finite nonnegative price")
        object.__setattr__(self, "quoted_price_hex", parsed_price.hex())

    @classmethod
    def from_values(
        cls,
        *,
        request_id: str,
        buyer_id: str,
        session_id: str,
        asset_id: str,
        sql: str,
        quoted_price: float,
        currency: str = "EXPERIMENTAL_PRICE_UNIT",
    ) -> "RequestContext":
        return cls(
            request_id=request_id,
            buyer_id=buyer_id,
            session_id=session_id,
            asset_id=asset_id,
            query_digest=exact_sql_digest(sql),
            quoted_price_hex=_float64_hex(quoted_price, nonnegative=True),
            currency=currency,
        )

    @property
    def quoted_price(self) -> float:
        return float.fromhex(self.quoted_price_hex)

    def binding_fields(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "buyer_id": self.buyer_id,
            "session_id": self.session_id,
            "asset_id": self.asset_id,
            "query_digest": self.query_digest,
            "quoted_price_hex": self.quoted_price_hex,
            "currency": self.currency,
        }

    @property
    def request_digest(self) -> str:
        return _sha256_hex(
            b"ARBITER-REQUEST-V2\x00",
            _canonical_json(self.binding_fields()),
        )


class BoundAnswerBuffer:
    """Immutable serialized answer plus mutable local release status.

    The payload bytes, buffer identifier, and binding digests are exposed only
    through read-only properties.  The durable SQLite row remains authoritative
    after a successful commit, allowing recovery if the process exits before its
    local ``released`` flag is observed.
    """

    __slots__ = (
        "_buffer_id",
        "_request_digest",
        "_answer_digest",
        "_payload_bytes",
        "_released",
        "_retained_reason",
    )
    _IMMUTABLE_SLOTS = {
        "_buffer_id",
        "_request_digest",
        "_answer_digest",
        "_payload_bytes",
    }

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._IMMUTABLE_SLOTS and hasattr(self, name):
            raise AttributeError(f"{name[1:]} is immutable after buffer creation")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        request: RequestContext,
        payload: Any,
        buffer_id: str | None = None,
    ) -> None:
        payload_bytes = canonical_answer_bytes(payload)
        self._buffer_id = _validated_identifier(
            "buffer_id", buffer_id if buffer_id is not None else secrets.token_hex(16)
        )
        self._request_digest = request.request_digest
        self._answer_digest = answer_digest(payload_bytes)
        self._payload_bytes = payload_bytes
        self._released = False
        self._retained_reason = "buffered"

    @property
    def buffer_id(self) -> str:
        return self._buffer_id

    @property
    def request_digest(self) -> str:
        return self._request_digest

    @property
    def answer_digest(self) -> str:
        return self._answer_digest

    @property
    def payload_bytes(self) -> bytes:
        return self._payload_bytes

    @property
    def released(self) -> bool:
        return self._released

    @property
    def retained_reason(self) -> str:
        return self._retained_reason

    def _mark_released(self) -> None:
        object.__setattr__(self, "_released", True)
        object.__setattr__(self, "_retained_reason", "")

    def retain(self, reason: str) -> None:
        # Release is monotonic. A replay attempt against the same local object
        # must not make an already delivered answer appear buffered again.
        if self._released:
            return
        object.__setattr__(self, "_released", False)
        object.__setattr__(
            self,
            "_retained_reason",
            _validated_identifier("retained_reason", reason),
        )


@dataclass(frozen=True)
class RequestBoundReleaseToken:
    protocol_version: int
    purpose: str
    token_id: str
    nonce: str
    issued_at_ns: int
    expires_at_ns: int
    epoch: int
    constraint_digest: str
    asset_digest: str
    price_digest: str
    verifier_tolerance_hex: str
    request_id: str
    buyer_id: str
    session_id: str
    asset_id: str
    query_digest: str
    quoted_price_hex: str
    currency: str
    request_digest: str
    buffer_id: str
    answer_digest: str
    mac: str

    def signed_fields(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "purpose": self.purpose,
            "token_id": self.token_id,
            "nonce": self.nonce,
            "issued_at_ns": self.issued_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "epoch": self.epoch,
            "constraint_digest": self.constraint_digest,
            "asset_digest": self.asset_digest,
            "price_digest": self.price_digest,
            "verifier_tolerance_hex": self.verifier_tolerance_hex,
            "request_id": self.request_id,
            "buyer_id": self.buyer_id,
            "session_id": self.session_id,
            "asset_id": self.asset_id,
            "query_digest": self.query_digest,
            "quoted_price_hex": self.quoted_price_hex,
            "currency": self.currency,
            "request_digest": self.request_digest,
            "buffer_id": self.buffer_id,
            "answer_digest": self.answer_digest,
        }


@dataclass(frozen=True)
class RequestBoundCommitResult:
    released: bool
    reason: str
    epoch: int
    token_id: str
    durable: bool


class DurableReleaseLedger:
    """SQLite-backed staged-answer outbox and release-authorization ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._initialize_schema()

    def _new_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._new_connection()
            self._local.connection = connection
        return connection

    def close_thread_connection(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _initialize_schema(self) -> None:
        with self._schema_lock:
            connection = self._new_connection()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS answer_buffers (
                        buffer_id TEXT PRIMARY KEY,
                        request_digest TEXT NOT NULL,
                        answer_digest TEXT NOT NULL,
                        payload BLOB NOT NULL,
                        state TEXT NOT NULL CHECK (state IN ('buffered', 'released')),
                        staged_at_ns INTEGER NOT NULL,
                        released_at_ns INTEGER,
                        token_id TEXT UNIQUE
                    );

                    CREATE TABLE IF NOT EXISTS release_commits (
                        token_id TEXT PRIMARY KEY,
                        request_id TEXT NOT NULL UNIQUE,
                        buyer_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        asset_id TEXT NOT NULL,
                        query_digest TEXT NOT NULL,
                        quoted_price_hex TEXT NOT NULL,
                        currency TEXT NOT NULL,
                        request_digest TEXT NOT NULL,
                        buffer_id TEXT NOT NULL UNIQUE,
                        answer_digest TEXT NOT NULL,
                        price_digest TEXT NOT NULL,
                        constraint_digest TEXT NOT NULL,
                        asset_digest TEXT NOT NULL,
                        epoch INTEGER NOT NULL,
                        verifier_tolerance_hex TEXT NOT NULL,
                        committed_at_ns INTEGER NOT NULL,
                        FOREIGN KEY(buffer_id) REFERENCES answer_buffers(buffer_id)
                    );

                    CREATE INDEX IF NOT EXISTS release_commits_buyer_request
                    ON release_commits(buyer_id, request_id);
                    """
                )
            finally:
                connection.close()

    def stage_buffer(self, request: RequestContext, buffer: BoundAnswerBuffer) -> None:
        if type(buffer) is not BoundAnswerBuffer:
            raise TypeError("buffer must be an exact BoundAnswerBuffer")
        if buffer.request_digest != request.request_digest:
            raise ValueError("buffer_request_digest_mismatch")
        # BoundAnswerBuffer computes this digest in its constructor and exposes
        # immutable ``bytes`` through read-only slots. Avoid hashing a large
        # answer twice before it can be staged.
        connection = self._connection()
        now_ns = time.time_ns()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT request_digest, answer_digest, state
                FROM answer_buffers WHERE buffer_id = ?
                """,
                (buffer.buffer_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO answer_buffers(
                        buffer_id, request_digest, answer_digest, payload,
                        state, staged_at_ns
                    ) VALUES (?, ?, ?, ?, 'buffered', ?)
                    """,
                    (
                        buffer.buffer_id,
                        buffer.request_digest,
                        buffer.answer_digest,
                        sqlite3.Binary(buffer.payload_bytes),
                        now_ns,
                    ),
                )
            else:
                identical = (
                    existing["request_digest"] == buffer.request_digest
                    and existing["answer_digest"] == buffer.answer_digest
                    and existing["state"] == "buffered"
                )
                if not identical:
                    raise ValueError("buffer_id_collision_or_already_released")
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def commit_release(
        self,
        token: RequestBoundReleaseToken,
        request: RequestContext,
        buffer: BoundAnswerBuffer,
        *,
        committed_at_ns: int,
    ) -> tuple[bool, str]:
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM release_commits WHERE token_id = ?",
                (token.token_id,),
            ).fetchone() is not None:
                connection.execute("ROLLBACK")
                return False, "token_replay"
            if connection.execute(
                "SELECT 1 FROM release_commits WHERE request_id = ?",
                (request.request_id,),
            ).fetchone() is not None:
                connection.execute("ROLLBACK")
                return False, "request_replay"

            # The answer BLOB was hashed and validated when staged.  Commit reads
            # only its immutable identity fields, keeping the authorization hot
            # path independent of answer size. Delivery revalidates the BLOB.
            staged = connection.execute(
                """
                SELECT request_digest, answer_digest, state
                FROM answer_buffers WHERE buffer_id = ?
                """,
                (buffer.buffer_id,),
            ).fetchone()
            if staged is None:
                connection.execute("ROLLBACK")
                return False, "buffer_not_staged"
            if staged["state"] != "buffered":
                connection.execute("ROLLBACK")
                return False, "buffer_already_released"
            if staged["request_digest"] != request.request_digest:
                connection.execute("ROLLBACK")
                return False, "ledger_request_digest_mismatch"
            if staged["answer_digest"] != buffer.answer_digest:
                connection.execute("ROLLBACK")
                return False, "ledger_answer_digest_mismatch"
            connection.execute(
                """
                INSERT INTO release_commits(
                    token_id, request_id, buyer_id, session_id, asset_id,
                    query_digest, quoted_price_hex, currency, request_digest,
                    buffer_id, answer_digest, price_digest, constraint_digest,
                    asset_digest, epoch, verifier_tolerance_hex, committed_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token.token_id,
                    request.request_id,
                    request.buyer_id,
                    request.session_id,
                    request.asset_id,
                    request.query_digest,
                    request.quoted_price_hex,
                    request.currency,
                    request.request_digest,
                    buffer.buffer_id,
                    buffer.answer_digest,
                    token.price_digest,
                    token.constraint_digest,
                    token.asset_digest,
                    token.epoch,
                    token.verifier_tolerance_hex,
                    int(committed_at_ns),
                ),
            )
            updated = connection.execute(
                """
                UPDATE answer_buffers
                SET state = 'released', released_at_ns = ?, token_id = ?
                WHERE buffer_id = ? AND state = 'buffered'
                """,
                (int(committed_at_ns), token.token_id, buffer.buffer_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError("durable buffer state transition was not unique")
            connection.execute("COMMIT")
            return True, "released"
        except sqlite3.IntegrityError as exc:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            message = str(exc).lower()
            if "token_id" in message:
                return False, "token_replay"
            if "request_id" in message:
                return False, "request_replay"
            if "buffer_id" in message:
                return False, "buffer_already_released"
            return False, "ledger_integrity_failure"
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def released_payload(
        self,
        *,
        token_id: str,
        request: RequestContext,
    ) -> bytes:
        token_id = _validated_identifier("token_id", token_id)
        connection = self._connection()
        row = connection.execute(
            """
            SELECT c.request_digest, c.answer_digest, b.payload, b.state
            FROM release_commits AS c
            JOIN answer_buffers AS b ON b.buffer_id = c.buffer_id
            WHERE c.token_id = ?
            """,
            (token_id,),
        ).fetchone()
        if row is None or row["state"] != "released":
            raise KeyError("released answer not found")
        if row["request_digest"] != request.request_digest:
            raise PermissionError("delivery_request_mismatch")
        payload = bytes(row["payload"])
        if row["answer_digest"] != answer_digest(payload):
            raise RuntimeError("durable_answer_integrity_failure")
        return payload

    def release_rows(self) -> list[dict[str, Any]]:
        connection = self._connection()
        rows = connection.execute(
            "SELECT * FROM release_commits ORDER BY committed_at_ns, token_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def buffer_rows(self) -> list[dict[str, Any]]:
        connection = self._connection()
        rows = connection.execute(
            """
            SELECT buffer_id, request_digest, answer_digest, length(payload) AS payload_bytes,
                   state, staged_at_ns, released_at_ns, token_id
            FROM answer_buffers ORDER BY staged_at_ns, buffer_id
            """
        ).fetchall()
        return [dict(row) for row in rows]


class RequestBoundSnapshotStore:
    """Pinned-v4 verifier plus a request-bound, durable compare-and-commit gate."""

    def __init__(
        self,
        initial_snapshot: _core.ConstraintSnapshot,
        *,
        ledger: DurableReleaseLedger,
        asset_query_digests: Mapping[str, str],
        secret: bytes | None = None,
        verifier_tolerance: float = _core.DEFAULT_TOLERANCE,
        token_ttl_ns: int = DEFAULT_TOKEN_TTL_NS,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        _core._validate_snapshot_identity(initial_snapshot)
        if not isinstance(ledger, DurableReleaseLedger):
            raise TypeError("ledger must be a DurableReleaseLedger")
        self._snapshot = initial_snapshot
        self._numeric_integrity_digest = _core._numeric_snapshot_integrity_digest(
            initial_snapshot
        )
        self._asset_index = {
            asset_id: index for index, asset_id in enumerate(initial_snapshot.asset_ids)
        }
        supplied_query_digests = {
            _validated_identifier("asset_id", str(asset_id)): _validated_digest(
                "query_digest", digest
            )
            for asset_id, digest in asset_query_digests.items()
        }
        if set(supplied_query_digests) != set(initial_snapshot.asset_ids):
            raise ValueError(
                "asset_query_digests must cover exactly the snapshot asset IDs"
            )
        self._asset_query_digests = supplied_query_digests
        self._ledger = ledger
        self._secret = secret if secret is not None else secrets.token_bytes(32)
        if not isinstance(self._secret, bytes) or len(self._secret) < 32:
            raise ValueError("secret must contain at least 32 bytes")
        self._verifier_tolerance = _validated_tolerance(verifier_tolerance)
        if isinstance(token_ttl_ns, bool) or int(token_ttl_ns) != token_ttl_ns or int(token_ttl_ns) <= 0:
            raise ValueError("token_ttl_ns must be a positive integer")
        self._token_ttl_ns = int(token_ttl_ns)
        self._clock_ns = clock_ns
        self._lock = threading.RLock()

    @property
    def ledger(self) -> DurableReleaseLedger:
        return self._ledger

    def read(self) -> _core.ConstraintSnapshot:
        with self._lock:
            return self._snapshot

    def advance(self, snapshot: _core.ConstraintSnapshot) -> None:
        with self._lock:
            if snapshot.epoch <= self._snapshot.epoch:
                raise ValueError("new snapshot epoch must increase")
            _core._validate_snapshot_identity(snapshot)
            if snapshot.asset_ids != self._snapshot.asset_ids:
                raise ValueError(
                    "asset registry changes require a new store and query-digest map"
                )
            self._snapshot = snapshot
            self._numeric_integrity_digest = _core._numeric_snapshot_integrity_digest(snapshot)
            self._asset_index = {
                asset_id: index for index, asset_id in enumerate(snapshot.asset_ids)
            }

    def restart_with_new_secret(self) -> None:
        with self._lock:
            self._secret = secrets.token_bytes(32)

    def _token_mac(self, signed_fields: Mapping[str, Any]) -> str:
        return hmac.new(
            self._secret,
            _canonical_json(dict(signed_fields)),
            hashlib.sha256,
        ).hexdigest()

    def _reject(
        self,
        reason: str,
        token: RequestBoundReleaseToken,
        buffer: BoundAnswerBuffer,
    ) -> RequestBoundCommitResult:
        buffer.retain(reason)
        return RequestBoundCommitResult(
            released=False,
            reason=reason,
            epoch=self._snapshot.epoch,
            token_id=getattr(token, "token_id", ""),
            durable=False,
        )

    def issue_token(
        self,
        snapshot: _core.ConstraintSnapshot,
        prices: np.ndarray,
        request: RequestContext,
        buffer: BoundAnswerBuffer,
        *,
        tolerance: float | None = None,
    ) -> RequestBoundReleaseToken:
        policy_tolerance = self._verifier_tolerance if tolerance is None else _validated_tolerance(tolerance)
        if not _same_float64(policy_tolerance, self._verifier_tolerance):
            raise ValueError("requested tolerance does not match the store policy")
        if type(request) is not RequestContext:
            raise TypeError("request must be an exact RequestContext")
        if type(buffer) is not BoundAnswerBuffer:
            raise TypeError("buffer must be an exact BoundAnswerBuffer")
        verification = _core.independent_verify(
            snapshot,
            prices,
            ordered_asset_ids=snapshot.asset_ids,
            tolerance=policy_tolerance,
        )
        if not verification.passed:
            raise ValueError("cannot issue a release token for an unverified vector")
        asset_index = {asset_id: index for index, asset_id in enumerate(snapshot.asset_ids)}
        if request.asset_id not in asset_index:
            raise ValueError("selected asset is absent from the snapshot")
        expected_query_digest = self._asset_query_digests.get(request.asset_id)
        if request.query_digest != expected_query_digest:
            raise ValueError("query digest does not match the selected asset registry")
        quoted_from_vector = float(np.asarray(prices, dtype=np.float64)[asset_index[request.asset_id]])
        if not _same_float64(request.quoted_price, quoted_from_vector):
            raise ValueError("quoted price does not equal the selected verified vector coordinate")
        if buffer.request_digest != request.request_digest:
            raise ValueError("answer buffer is not bound to this request")
        self._ledger.stage_buffer(request, buffer)

        issued_at_ns = int(self._clock_ns())
        fields = {
            "protocol_version": PROTOCOL_VERSION,
            "purpose": PROTOCOL_PURPOSE,
            "token_id": secrets.token_hex(16),
            "nonce": secrets.token_hex(16),
            "issued_at_ns": issued_at_ns,
            "expires_at_ns": issued_at_ns + self._token_ttl_ns,
            "epoch": int(snapshot.epoch),
            "constraint_digest": snapshot.constraint_digest,
            "asset_digest": snapshot.asset_digest,
            "price_digest": _core.price_vector_digest(snapshot.asset_ids, prices),
            "verifier_tolerance_hex": _float64_hex(policy_tolerance),
            **request.binding_fields(),
            "request_digest": request.request_digest,
            "buffer_id": buffer.buffer_id,
            "answer_digest": buffer.answer_digest,
        }
        return RequestBoundReleaseToken(mac=self._token_mac(fields), **fields)

    def compare_and_commit(
        self,
        token: RequestBoundReleaseToken,
        prices: np.ndarray,
        request: RequestContext,
        buffer: BoundAnswerBuffer,
    ) -> RequestBoundCommitResult:
        if type(token) is not RequestBoundReleaseToken:
            raise TypeError("token must be an exact RequestBoundReleaseToken")
        if type(request) is not RequestContext:
            raise TypeError("request must be an exact RequestContext")
        if type(buffer) is not BoundAnswerBuffer:
            raise TypeError("buffer must be an exact BoundAnswerBuffer")

        with self._lock:
            current = self._snapshot
            if _core._numeric_snapshot_integrity_digest(current) != self._numeric_integrity_digest:
                return self._reject("snapshot_integrity_failure", token, buffer)
            if token.protocol_version != PROTOCOL_VERSION or token.purpose != PROTOCOL_PURPOSE:
                return self._reject("protocol_mismatch", token, buffer)
            try:
                expected_mac = self._token_mac(token.signed_fields())
                token_tolerance = float.fromhex(token.verifier_tolerance_hex)
                token_quote = float.fromhex(token.quoted_price_hex)
                if not math.isfinite(token_tolerance) or token_tolerance <= 0.0:
                    raise ValueError("invalid tolerance")
                if not math.isfinite(token_quote) or token_quote < 0.0:
                    raise ValueError("invalid quote")
            except Exception:
                return self._reject("invalid_token_fields", token, buffer)
            if not hmac.compare_digest(token.mac, expected_mac):
                return self._reject("token_mac_mismatch", token, buffer)
            now_ns = int(self._clock_ns())
            if now_ns > token.expires_at_ns:
                return self._reject("token_expired", token, buffer)
            if token.issued_at_ns > now_ns:
                return self._reject("token_from_future", token, buffer)
            if token.epoch != current.epoch:
                return self._reject("stale_epoch", token, buffer)
            if token.constraint_digest != current.constraint_digest:
                return self._reject("constraint_digest_mismatch", token, buffer)
            if token.asset_digest != current.asset_digest:
                return self._reject("asset_digest_mismatch", token, buffer)
            if not _same_float64(token_tolerance, self._verifier_tolerance):
                return self._reject("tolerance_policy_mismatch", token, buffer)

            expected_request_fields = request.binding_fields()
            for name, expected in expected_request_fields.items():
                if getattr(token, name) != expected:
                    return self._reject(f"{name}_mismatch", token, buffer)
            if token.request_digest != request.request_digest:
                return self._reject("request_digest_mismatch", token, buffer)
            if buffer.request_digest != request.request_digest:
                return self._reject("buffer_request_mismatch", token, buffer)
            if token.buffer_id != buffer.buffer_id:
                return self._reject("buffer_id_mismatch", token, buffer)
            if token.answer_digest != buffer.answer_digest:
                return self._reject("answer_digest_mismatch", token, buffer)

            vector = np.asarray(prices, dtype=np.float64)
            if token.asset_id not in self._asset_index:
                return self._reject("selected_asset_missing", token, buffer)
            if token.query_digest != self._asset_query_digests[token.asset_id]:
                return self._reject("asset_query_registry_mismatch", token, buffer)
            actual_price_digest = _core.price_vector_digest(current.asset_ids, vector)
            if token.price_digest != actual_price_digest:
                return self._reject("price_digest_mismatch", token, buffer)
            quote_from_vector = float(vector[self._asset_index[token.asset_id]])
            if not _same_float64(token_quote, quote_from_vector):
                return self._reject("quote_vector_mismatch", token, buffer)
            verification = _core.independent_verify(
                current,
                vector,
                ordered_asset_ids=current.asset_ids,
                tolerance=token_tolerance,
            )
            if not verification.passed:
                return self._reject("commit_verification_failed", token, buffer)

            durable, reason = self._ledger.commit_release(
                token,
                request,
                buffer,
                committed_at_ns=now_ns,
            )
            if not durable:
                return self._reject(reason, token, buffer)
            buffer._mark_released()
            return RequestBoundCommitResult(
                released=True,
                reason="released",
                epoch=current.epoch,
                token_id=token.token_id,
                durable=True,
            )


def replace_bound_token(
    token: RequestBoundReleaseToken,
    **changes: Any,
) -> RequestBoundReleaseToken:
    """Fault-injection helper; changed signed fields invalidate the MAC."""

    return replace(token, **changes)


def next_epoch_snapshot(snapshot: _core.ConstraintSnapshot) -> _core.ConstraintSnapshot:
    """Construct an identity-equivalent snapshot at the next epoch for tests."""

    return _core.ConstraintSnapshot(
        asset_ids=snapshot.asset_ids,
        matrix=snapshot.matrix,
        bounds=snapshot.bounds,
        row_metadata=snapshot.row_metadata,
        epoch=snapshot.epoch + 1,
        constraint_digest=snapshot.constraint_digest,
        asset_digest=snapshot.asset_digest,
    )
