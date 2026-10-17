"""Shared utilities: timing, logging, JSONL I/O, provenance, small helpers.

Everything here is dependency-light (stdlib + numpy) so it is importable in a
bare environment and in tests.  No module in FactAnchor may print() -- use
:func:`get_logger`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

__all__ = [
    "Timer",
    "TimerResult",
    "get_logger",
    "setup_logging",
    "flush_all_handlers",
    "sha256_text",
    "sha256_file",
    "json_default",
    "write_json",
    "read_json",
    "append_jsonl",
    "write_jsonl",
    "read_jsonl",
    "iter_jsonl",
    "ensure_dir",
    "resolve_run_dir",
    "new_run_id",
    "git_info",
    "collect_environment",
    "library_versions",
    "set_global_seed",
    "chunked",
    "safe_ratio",
    "truncate",
    "slugify",
    "RateLimiter",
    "DiskCache",
    "JsonlWriter",
    "NullContext",
]

T = TypeVar("T")

LOGGER_NAME = "factanchor"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger (``factanchor.<name>``)."""
    full = LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(full)


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: Path | str | None = None,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the ``factanchor`` logger once; safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        if not quiet:
            stream = logging.StreamHandler(stream=sys.stderr)
            stream.setFormatter(formatter)
            logger.addHandler(stream)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        already = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == Path(resolved)
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def flush_all_handlers() -> None:
    """Flush every handler of the project logger (call before reading log files)."""
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        try:
            handler.flush()
        except Exception:  # pragma: no cover - handler dependent
            pass


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TimerResult:
    """Mutable accumulator of elapsed wall-clock time for one label."""

    __slots__ = ("label", "count", "total")

    def __init__(self, label: str) -> None:
        self.label = label
        self.count = 0
        self.total = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"TimerResult({self.label!r}, n={self.count}, total={self.total:.6f}, mean={self.mean:.6f})"


class Timer:
    """Wall-clock timer using :func:`time.monotonic`.

    Synchronisation policy (spec 10.4): when ``synchronize=True`` and CUDA is
    available, ``torch.cuda.synchronize()`` is called on entry and exit so that
    asynchronous GPU kernels cannot produce misleadingly small durations.  Torch
    is imported lazily and its absence simply disables synchronisation.
    """

    def __init__(self, label: str = "timer", *, synchronize: bool = False) -> None:
        self.label = label
        self.synchronize = synchronize
        self._start: float | None = None
        self.elapsed: float = 0.0
        self.result = TimerResult(label)

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop(record=True)

    def start(self) -> Timer:
        self._sync()
        self._start = time.monotonic()
        return self

    def stop(self, *, record: bool = True) -> float:
        if self._start is None:
            raise RuntimeError(f"Timer {self.label!r} was stopped before it was started")
        self._sync()
        self.elapsed = time.monotonic() - self._start
        self._start = None
        if record:
            self.result.count += 1
            self.result.total += self.elapsed
        return self.elapsed

    @staticmethod
    def _sync() -> None:
        try:  # pragma: no cover - depends on torch presence
            import torch

            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return


@contextmanager
def timed_torch_call() -> Iterator[None]:
    """Alias kept for readability at call sites (same semantics as ``Timer``)."""
    timer = Timer(label="torch_call", synchronize=True)
    with timer:
        yield


# ---------------------------------------------------------------------------
# Hashing / JSON I/O
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """JSON encoder fallback for Path/NumPy/objects with ``to_dict``."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except TypeError:  # pragma: no cover - signature mismatch
            pass
    try:
        import numpy as np

        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # pragma: no cover - numpy optional
        pass
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Atomically write JSON (temp file + replace) so partial writes cannot happen."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent, default=json_default)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    return target


def read_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path | str, records: Iterable[Any], *, flush: bool = True) -> Path:
    """Append records (dicts or pydantic models) as JSON lines."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(_jsonl_line(record))
        if flush:
            handle.flush()
            os.fsync(handle.fileno())
    return target


def write_jsonl(path: Path | str, records: Iterable[Any], *, overwrite: bool = True) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists; refusing to overwrite (pass overwrite=True explicitly)"
        )
    with open(target, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_jsonl_line(record))
    return target


def _jsonl_line(record: Any) -> str:
    if hasattr(record, "model_dump_json"):
        payload = record.model_dump(mode="json")
    elif isinstance(record, dict):
        payload = record
    else:
        raise TypeError(f"cannot serialise {type(record).__name__} as a JSON line")
    return json.dumps(payload, ensure_ascii=False, default=json_default) + "\n"


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield parsed objects; raises with the offending line number on bad JSON."""
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{lineno}: {exc}") from exc


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


class JsonlWriter:
    """Append-only JSONL writer that can buffer records for throughput."""

    def __init__(self, path: Path | str, *, buffer_size: int = 0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = max(0, int(buffer_size))
        self._buffer: list[str] = []
        self._handle = open(self.path, "a", encoding="utf-8")

    def write(self, record: Any) -> None:
        self._buffer.append(_jsonl_line(record))
        if len(self._buffer) >= max(1, self.buffer_size):
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self._handle.writelines(self._buffer)
            self._buffer.clear()
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self.flush()
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Paths / run directories
# ---------------------------------------------------------------------------


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def slugify(text: str, *, max_length: int = 40) -> str:
    keep = [ch.lower() if ch.isalnum() else "-" for ch in text.strip()]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:max_length] or "run"


def new_run_id(name: str = "run", *, when: float | None = None) -> str:
    """Timestamped, sortable, human-readable run id: ``20261011T101530Z-name-ab12cd``."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(when if when is not None else time.time()))
    entropy = f"{random.SystemRandom().randint(0, 0xFFFFFF):06x}"
    return f"{stamp}-{slugify(name)}-{entropy}"


def resolve_run_dir(
    base_dir: Path | str,
    run_id: str,
    *,
    overwrite: bool = False,
    allow_reuse: bool = False,
) -> tuple[Path, list[str]]:
    """Create ``base_dir/runs/<run_id>`` without ever clobbering results.

    Returns ``(path, warnings)``.  If the directory exists:

    * ``overwrite=True``  -> reuse it, warning that previous results may be
      mixed with new ones (only the explicit flag permits this);
    * ``allow_reuse=True`` -> reuse it (used for resuming a run);
    * otherwise a unique suffix is appended so nothing is lost.
    """
    base = Path(base_dir) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    target = base / run_id
    if target.exists():
        if overwrite:
            warnings.append(f"run directory {target} reused with run.overwrite=true; results may mix runs")
        elif allow_reuse:
            warnings.append(f"run directory {target} reused (allow_reuse=true); appending to previous artifacts")
        else:
            suffix = 1
            while (base / f"{run_id}-{suffix}").exists():
                suffix += 1
            new_target = base / f"{run_id}-{suffix}"
            warnings.append(
                f"run directory {target} already exists; writing to {new_target} instead of overwriting"
            )
            target = new_target
    for sub in ("plots",):
        (target / sub).mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    return target, warnings


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_info(cwd: Path | str | None = None) -> dict[str, Any]:
    """Best-effort git commit hash + dirty flag.  Never raises."""
    info: dict[str, Any] = {"hash": None, "dirty": None, "branch": None}
    try:
        run = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if run.returncode == 0:
            info["hash"] = run.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        pass
    return info


def library_versions(packages: Sequence[str] | None = None) -> dict[str, str]:
    """Versions of relevant libraries (missing packages are recorded as absent)."""
    from importlib import metadata

    wanted = list(
        packages
        or [
            "torch",
            "transformers",
            "tokenizers",
            "datasets",
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "pydantic",
            "pyyaml",
            "requests",
            "spacy",
            "streamlit",
            "bert-score",
            "pytest",
        ]
    )
    out: dict[str, str] = {}
    for name in wanted:
        try:
            out[name] = metadata.version(name)
        except Exception:
            out[name] = "not-installed"
    return out


def collect_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full environment snapshot for ``environment.json`` (spec 16)."""
    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "library_versions": library_versions(),
    }
    # thread-pool sizes matter for CPU latency measurements
    env["threads"] = {
        "torch_num_threads": None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    try:  # pragma: no cover - torch optional
        import torch

        env["threads"]["torch_num_threads"] = torch.get_num_threads()
        env["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except Exception:
        env["torch"] = {"version": "not-installed"}
    try:  # pragma: no cover - psutil optional
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        env["peak_rss_mb"] = usage.ru_maxrss / 1024.0 if sys.platform != "darwin" else usage.ru_maxrss / (1024.0 * 1024.0)
    except Exception:
        env["peak_rss_mb"] = None
    if extra:
        env.update(extra)
    return env


def set_global_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed python/numpy/torch and (optionally) enable deterministic algorithms.

    Returns a small report so the run manifest can record what was actually
    applied (``torch`` absent is not an error).
    """
    report: dict[str, Any] = {"seed": seed, "deterministic_algorithms": False, "torch_seeded": False}
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # pragma: no cover - numpy optional
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                report["deterministic_algorithms"] = True
            except Exception:
                report["deterministic_algorithms"] = False
        report["torch_seeded"] = True
    except Exception:
        pass
    return report


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """``numerator/denominator`` or ``None`` when the denominator is zero.

    Returning ``None`` (rather than 0.0) keeps "no claims" distinguishable from
    "all claims wrong" in the metric tables.
    """
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def truncate(text: str, max_chars: int, *, ellipsis: str = "...") -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(ellipsis))] + ellipsis


class NullContext:
    """A no-op context manager (used when a timer/collector is disabled)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: object) -> bool:
        return False


class RateLimiter:
    """Thread-safe minimum-interval limiter used for knowledge API calls."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        self.min_interval = 1.0 / float(requests_per_second)
        self._lock = threading.Lock()
        self._last: float = 0.0
        self.sleeps = 0

    def acquire(self) -> float:
        """Block until the next request is allowed; returns the slept seconds."""
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
                self.sleeps += 1
                self._last = time.monotonic()
            else:
                self._last = now
            return max(0.0, wait)


class DiskCache:
    """Tiny JSON file cache keyed by a namespace + a hash of the request.

    Used for entity linking and fact retrieval so that repeated runs (and the
    demo) do not hammer the knowledge API.  Values carry a UTC timestamp; entries
    older than ``ttl_seconds`` (0 = never expire) are ignored.
    """

    def __init__(self, root: Path | str, *, ttl_seconds: int = 0, enabled: bool = True) -> None:
        self.root = Path(root)
        self.ttl_seconds = int(ttl_seconds)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{sha256_text(key)[:32]}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        if self.ttl_seconds > 0:
            stored_at = float(payload.get("stored_at", 0.0))
            if time.time() - stored_at > self.ttl_seconds:
                self.misses += 1
                return None
        self.hits += 1
        return payload.get("value")

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_json(path, {"stored_at": time.time(), "key": key, "value": value}, indent=None)
        except OSError:  # pragma: no cover - disk full / permissions
            return

    def clear(self, namespace: str | None = None) -> int:
        """Delete cached files; returns the number removed."""
        target = self.root / namespace if namespace else self.root
        removed = 0
        if not target.exists():
            return 0
        for path in sorted(target.rglob("*.json")):
            try:
                path.unlink()
                removed += 1
            except OSError:  # pragma: no cover
                pass
        return removed

    def stats(self) -> dict[str, int | bool]:
        return {"enabled": self.enabled, "hits": self.hits, "misses": self.misses, "ttl_seconds": self.ttl_seconds}

    def fingerprint(self, namespace: str, key: str, value: Any) -> str:
        """Deterministic value hash, handy to assert cache correctness in tests."""
        return sha256_text(json.dumps([namespace, key, value], sort_keys=True, default=json_default))


def batched(items: list[T], batch_size: int) -> list[list[T]]:
    """Split ``items`` into consecutive batches (batch_size must be >= 1)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
