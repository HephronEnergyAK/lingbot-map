"""Native Windows integration for the catalogued SkySeg CPU execution path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER_SOURCE = ROOT / "worker" / "src"
if str(WORKER_SOURCE) not in sys.path:
    sys.path.insert(0, str(WORKER_SOURCE))

from lingbot_map_worker.canonical_preprocessing import canonicalize_srgb
from lingbot_map_worker.fixture_job import _install_audit_policy
from lingbot_map_worker.sky_masking import SkyMaskRequest, SkyMaskSession


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _request(managed_root: Path, model_path: Path) -> SkyMaskRequest:
    return SkyMaskRequest(
        managed_root=managed_root,
        source_sha256=hashlib.sha256(b"native-windows-sky-mask-fixture").hexdigest(),
        video_stream_index=0,
        display_transform="identity",
        color_standard="bt709",
        color_range="limited",
        frame_count=2,
        model_grid_shape=(294, 518),
        model_id="skyseg",
        model_path=model_path,
        model_sha256=_sha256_file(model_path),
        worker_version="0.1.0",
        onnx_threads=1,
        cancel=lambda: False,
    )


def _frames() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 255, 1280, dtype=np.uint8)
    sky = np.broadcast_to(x[None, :], (720, 1280))
    first = np.stack((sky, np.full_like(sky, 160), 255 - sky), axis=2)
    second = np.flip(first, axis=1)
    return np.ascontiguousarray(first), np.ascontiguousarray(second)


def _run(
    request: SkyMaskRequest,
) -> tuple[SkyMaskSession, dict[str, object], tuple[float, ...]]:
    session = SkyMaskSession(request)
    fractions: list[float] = []
    for index, source in enumerate(_frames()):
        canonical = canonicalize_srgb(source)
        model_before = canonical.model_input.copy()
        color_before = canonical.color_rgb.copy()
        session.prepare(index, canonical)
        mask, fraction = session.mask_for(index, request.model_grid_shape)
        np.testing.assert_array_equal(canonical.model_input, model_before)
        np.testing.assert_array_equal(canonical.color_rgb, color_before)
        assert mask.dtype.str == "|u1"
        assert mask.shape == request.model_grid_shape
        assert mask.flags.c_contiguous
        assert bool(np.isin(mask, (0, 1)).all())
        assert np.isfinite(fraction) and 0 <= float(fraction) <= 1
        fractions.append(float(fraction))
    outcome = session.finish()
    np.testing.assert_allclose(outcome.sky_fraction, fractions, rtol=0, atol=1e-7)
    return session, dict(outcome.provenance), tuple(fractions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--managed-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    arguments = parser.parse_args()
    managed_root = arguments.managed_root.resolve()
    model_path = arguments.model.resolve()
    assert sys.platform == "win32", sys.platform
    assert model_path.is_relative_to(managed_root)
    assert model_path.stat().st_size == 175_997_079
    assert (
        _sha256_file(model_path)
        == "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39"
    )

    _install_audit_policy()
    request = _request(managed_root, model_path)
    first_session, first, fractions = _run(request)
    second_session, second, cached_fractions = _run(request)
    assert first["provider"] == "CPUExecutionProvider"
    assert first["onnxruntime_version"] == "1.23.2"
    assert first["batch_size"] == 1
    assert first["onnx_threads"] == 1
    assert first["cache_status"] == "generated"
    assert second["cache_status"] == "hit"
    assert first_session.cache_key == second_session.cache_key
    assert fractions == cached_fractions
    print(
        "LINGBOT_MAP_SKY_MASK_WINDOWS="
        + json.dumps(
            {
                "python": sys.version.split()[0],
                "provider": first["provider"],
                "onnxruntime": first["onnxruntime_version"],
                "batch_size": first["batch_size"],
                "onnx_threads": first["onnx_threads"],
                "frames": len(fractions),
                "fractions": fractions,
                "first_cache_status": first["cache_status"],
                "second_cache_status": second["cache_status"],
                "cache_key": first_session.cache_key,
                "model_sha256": request.model_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
