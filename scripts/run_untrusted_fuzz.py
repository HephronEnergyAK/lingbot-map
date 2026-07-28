"""Deterministic coverage-guided fuzz gate for bounded untrusted parsers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import tracemalloc
from types import ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
for source in (ROOT, WORKER_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from lingbot_map_worker import ipc as worker_ipc
from lingbot_map_worker import result_bundle


def _load_extension_ipc() -> Any:
    package_name = "lingbot_map_fuzz_extension"
    package = ModuleType(package_name)
    package.__path__ = [str(ROOT / "blender_extension")]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        package_name + ".ipc",
        ROOT / "blender_extension" / "ipc.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Extension IPC parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extension_ipc = _load_extension_ipc()


JSON_SEEDS = (
    b"{}",
    b"[]",
    b'{"schema_version":"1.0.0"}',
    b'{"duplicate":1,"duplicate":2}',
    b'{"value":NaN}',
    b'{"value":Infinity}',
    b"\xef\xbb\xbf{}",
    b"\xff\xfe{}",
    b'{"truncated":',
    b"[" * 40 + b"0" + b"]" * 40,
    b'{"path":"../outside"}',
    b'{"path":"C:/absolute"}',
    b'{"shape":[9223372036854775807,9223372036854775807]}',
    b'{"integer":' + b"9" * 5000 + b"}",
    b" " * (1024 * 1024 + 1),
)


def _mutate(source: bytes, randomizer: random.Random) -> bytes:
    data = bytearray(source)
    operation = randomizer.randrange(7)
    if operation == 0 and data:
        index = randomizer.randrange(len(data))
        data[index] ^= 1 << randomizer.randrange(8)
    elif operation == 1 and data:
        start = randomizer.randrange(len(data))
        stop = min(len(data), start + randomizer.randrange(1, 33))
        del data[start:stop]
    elif operation == 2:
        index = randomizer.randrange(len(data) + 1)
        token = randomizer.choice(
            (b"{}", b"[]", b"null", b"NaN", b"../", b"\\xff", b"9" * 64)
        )
        data[index:index] = token
    elif operation == 3 and data:
        start = randomizer.randrange(len(data))
        stop = min(len(data), start + randomizer.randrange(1, 33))
        index = randomizer.randrange(len(data) + 1)
        data[index:index] = data[start:stop]
    elif operation == 4:
        data = bytearray(b"[" + data + b"]")
    elif operation == 5:
        data = bytearray(b'{"fuzz":' + data + b"}")
    else:
        randomizer.shuffle(data)
    return bytes(data[: 1024 * 1024 + 1])


def _trace_call(
    function: Callable[[bytes], Any],
    data: bytes,
    files: frozenset[str],
) -> tuple[str, frozenset[tuple[str, int]], str | None]:
    lines: set[tuple[str, int]] = set()

    def trace(frame: Any, event: str, _argument: Any):
        filename = str(Path(frame.f_code.co_filename).resolve())
        if event == "line" and filename in files:
            lines.add((filename, frame.f_lineno))
        return trace

    started = time.perf_counter()
    sys.settrace(trace)
    try:
        function(data)
    except (extension_ipc.IpcError, worker_ipc.IpcError):
        outcome, detail = "reject", None
    except Exception as exc:
        outcome = "unexpected"
        detail = f"{type(exc).__name__}: {exc}"
    else:
        outcome, detail = "accept", None
    finally:
        sys.settrace(None)
    elapsed = time.perf_counter() - started
    if elapsed > 0.25:
        return "unexpected", frozenset(lines), f"case exceeded 0.25s: {elapsed:.6f}s"
    return outcome, frozenset(lines), detail


def _write_failure(
    failure_root: Path,
    index: int,
    data: bytes,
    detail: str,
    predicate: Callable[[bytes], bool],
) -> None:
    minimized = _minimize_failure(data, predicate)
    (failure_root / f"failure-{index:04d}.bin").write_bytes(minimized)
    (failure_root / f"failure-{index:04d}.txt").write_text(
        "\n".join(
            (
                detail,
                f"original_bytes={len(data)}",
                f"minimal_bytes={len(minimized)}",
                f"minimal_sha256={hashlib.sha256(minimized).hexdigest()}",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )


def _minimize_failure(
    data: bytes,
    predicate: Callable[[bytes], bool],
) -> bytes:
    """Return a deterministic one-minimal byte-deletion reproducer."""

    if not predicate(data):
        raise ValueError("failure predicate does not reproduce the input")
    current = data
    granularity = 2
    while current:
        chunk_size = max(1, (len(current) + granularity - 1) // granularity)
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate = current[:start] + current[start + chunk_size :]
            if predicate(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if chunk_size == 1:
            break
        granularity = min(len(current), granularity * 2)
    return current


def _fuzz_json(
    *,
    randomizer: random.Random,
    cases: int,
    failure_root: Path,
) -> dict[str, Any]:
    targets = (
        ("extension", extension_ipc.parse_json_bytes),
        ("worker", worker_ipc.parse_json_bytes),
    )
    traced_files = frozenset(
        str(Path(function.__code__.co_filename).resolve())
        for _name, function in targets
    )
    corpus = list(JSON_SEEDS)
    coverage: set[tuple[str, int]] = set()
    accepted = rejected = 0
    failures = 0
    for index in range(cases):
        data = corpus[index] if index < len(corpus) else _mutate(
            randomizer.choice(corpus), randomizer
        )
        outcomes = []
        new_lines: set[tuple[str, int]] = set()
        for name, target in targets:
            outcome, lines, detail = _trace_call(target, data, traced_files)
            outcomes.append(outcome)
            new_lines.update(lines)
            if outcome == "unexpected":
                failures += 1
                _write_failure(
                    failure_root,
                    failures,
                    data,
                    f"{name}: {detail}",
                    lambda candidate, target=target: _trace_call(
                        target,
                        candidate,
                        traced_files,
                    )[0]
                    == "unexpected",
                )
        if len(set(outcomes)) != 1:
            failures += 1
            _write_failure(
                failure_root,
                failures,
                data,
                f"Extension/Worker parser parity differed: {outcomes!r}",
                lambda candidate: len(
                    {
                        _trace_call(target, candidate, traced_files)[0]
                        for _name, target in targets
                    }
                )
                != 1,
            )
        if outcomes[0] == "accept":
            accepted += 1
        elif outcomes[0] == "reject":
            rejected += 1
        discovered = new_lines - coverage
        if discovered:
            coverage.update(discovered)
            if data not in corpus and len(corpus) < 256:
                corpus.append(data)
    return {
        "cases": cases,
        "accepted": accepted,
        "rejected": rejected,
        "coverage_lines": len(coverage),
        "coverage_corpus": len(corpus),
        "failures": failures,
    }


def _npy_seed() -> bytes:
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1, 3), }"
    padding = b" " * ((64 - (10 + len(header) + 1) % 64) % 64)
    document = header + padding + b"\n"
    return b"\x93NUMPY\x01\x00" + len(document).to_bytes(2, "little") + document


def _fuzz_npy(
    *,
    randomizer: random.Random,
    cases: int,
    failure_root: Path,
) -> dict[str, Any]:
    seeds = [
        _npy_seed(),
        b"",
        b"\x93NUMPY\x01\x00\xff\xff",
        b"\x93NUMPY\x02\x00\x01\x00\x01\x00",
        b"\x93NUMPY\x01\x00\x20\x00{'descr':'O'}",
    ]
    accepted = rejected = failures = 0
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "fuzz.npy"

        def outcome(candidate: bytes) -> tuple[str, str | None]:
            path.write_bytes(candidate[: 70 * 1024])
            started = time.perf_counter()
            try:
                result_bundle._npy_header(path)
            except result_bundle.ResultBundleError:
                state, detail = "reject", None
            except Exception as exc:
                state = "unexpected"
                detail = f"NPY: {type(exc).__name__}: {exc}"
            else:
                state, detail = "accept", None
            elapsed = time.perf_counter() - started
            if elapsed > 0.25:
                return "unexpected", f"NPY case exceeded 0.25s: {elapsed:.6f}s"
            return state, detail

        for index in range(cases):
            data = seeds[index] if index < len(seeds) else _mutate(
                randomizer.choice(seeds), randomizer
            )
            state, detail = outcome(data)
            if state == "reject":
                rejected += 1
            elif state == "unexpected":
                failures += 1
                _write_failure(
                    failure_root,
                    10_000 + failures,
                    data,
                    detail or "NPY parser failed unexpectedly",
                    lambda candidate: outcome(candidate)[0] == "unexpected",
                )
            else:
                accepted += 1
    return {
        "cases": cases,
        "accepted": accepted,
        "rejected": rejected,
        "failures": failures,
    }


def run(seed: int, cases: int, failure_root: Path) -> dict[str, Any]:
    failure_root.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    tracemalloc.start()
    started = time.perf_counter()
    json_result = _fuzz_json(
        randomizer=randomizer,
        cases=cases,
        failure_root=failure_root,
    )
    npy_result = _fuzz_npy(
        randomizer=randomizer,
        cases=max(100, cases // 4),
        failure_root=failure_root,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "schema_version": "1.0.0",
        "seed": seed,
        "json": json_result,
        "npy": npy_result,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_traced_bytes": peak,
        "failure_count": json_result["failures"] + npy_result["failures"],
    }
    if peak > 128 * 1024 * 1024:
        result["failure_count"] += 1
        result["memory_error"] = "peak traced memory exceeded 128 MiB"
    (failure_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--failure-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not 100 <= arguments.cases <= 100_000:
        parser.error("--cases must be in [100,100000]")
    result = run(arguments.seed, arguments.cases, arguments.failure_dir.resolve())
    print(
        "LINGBOT_MAP_UNTRUSTED_FUZZ="
        + json.dumps(result, sort_keys=True, separators=(",", ":"))
    )
    return 0 if result["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
