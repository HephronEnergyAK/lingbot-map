"""Build the licensed KITTI windowed release fixture from the official archive."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT = (
    ROOT / "release_corpus" / "kitti-odometry-00-source.json"
)
FIXTURE_ID = "real-kitti-odometry-00-windowed-3001"
ARCHIVE_FILENAME = "data_odometry_gray.zip"
ARCHIVE_BYTES = 23_166_524_501
ARCHIVE_ETAG = '"1129e5e7249778de3bc174315455a681-2762"'
ARCHIVE_LAST_MODIFIED = "Fri, 11 May 2018 16:05:50 GMT"
ARCHIVE_URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    "data_odometry_gray.zip"
)
DATASET_URL = "https://www.cvlibs.net/datasets/kitti/eval_odometry.php"
LICENSE_EVIDENCE_URL = "https://www.cvlibs.net/datasets/kitti/"
LICENSE_DEED_URL = (
    "https://creativecommons.org/licenses/by-nc-sa/3.0/"
)
LICENSE_EVIDENCE_PATH = (
    "release_corpus/licenses/KITTI-CC-BY-NC-SA-3.0.md"
)
LICENSE_EVIDENCE_SHA256 = (
    "52ec921de2a59b2028215d5cb10d6fcac584c7fca1b132989def0f886aeda871"
)
SEQUENCE = "00"
CAMERA = "image_0"
FIRST_FRAME = 0
LAST_FRAME = 3000
FRAME_COUNT = LAST_FRAME - FIRST_FRAME + 1
SOURCE_WIDTH = 1241
SOURCE_HEIGHT = 376
OUTPUT_WIDTH = 518
OUTPUT_HEIGHT = 158
OUTPUT_RATE = 10
MAX_SOURCE_PNG_BYTES = 4 * 1024 * 1024
SELECTION = {
    "sequence": SEQUENCE,
    "camera": CAMERA,
    "source_frame_range": [FIRST_FRAME, LAST_FRAME],
    "source_files": FRAME_COUNT,
    "source_dimensions": [SOURCE_WIDTH, SOURCE_HEIGHT],
    "output_dimensions": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
    "fps": OUTPUT_RATE,
    "temporal_sampling": False,
}
LICENSE = {
    "spdx": "CC-BY-NC-SA-3.0",
    "evidence_path": LICENSE_EVIDENCE_PATH,
    "evidence_sha256": LICENSE_EVIDENCE_SHA256,
    "official_evidence_url": LICENSE_EVIDENCE_URL,
    "license_deed_url": LICENSE_DEED_URL,
    "attribution": (
        "Andreas Geiger, Philip Lenz, Christoph Stiller, "
        "and Raquel Urtasun; KITTI Vision Benchmark Suite"
    ),
    "constraints": [
        "attribution-required",
        "non-commercial-only",
        "share-alike-3.0-required-for-derivatives",
    ],
}
SOURCE_CONTRACT_KEYS = {
    "schema_version",
    "fixture_id",
    "status",
    "source_archive",
    "selection",
    "license",
    "observed_evidence",
    "expected_output",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_release_corpus import (  # noqa: E402
    COLOR_CASES,
    _encode_video,
    _record,
)
from scripts.validate_release_corpus import (  # noqa: E402
    CorpusValidationError,
    load_json,
)


class KittiFixtureError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


def _identity(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
    )


def _ordinary_file(path: Path) -> bool:
    try:
        value = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(value, "st_file_attributes", 0))
    return (
        stat.S_ISREG(value.st_mode)
        and not path.is_symlink()
        and not bool(
            attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    )


def _empty_plain_directory(path: Path) -> Path:
    destination = Path(os.path.abspath(path))
    if destination.exists():
        attributes = int(
            getattr(destination.lstat(), "st_file_attributes", 0)
        )
        if (
            not destination.is_dir()
            or destination.is_symlink()
            or bool(
                attributes
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
            or any(destination.iterdir())
        ):
            raise KittiFixtureError(
                "KITTI fixture output must be a new or empty plain directory"
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = _identity(os.fstat(stream.fileno()))
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = _identity(os.fstat(stream.fileno()))
    if before != after or _identity(path.stat()) != after:
        raise KittiFixtureError(
            "KITTI archive changed while its checksum was computed"
        )
    return digest.hexdigest()


def _member_name(index: int) -> str:
    return (
        f"dataset/sequences/{SEQUENCE}/{CAMERA}/"
        f"{index:06d}.png"
    )


def _require_source_contract() -> dict[str, Any]:
    try:
        contract = load_json(SOURCE_CONTRACT)
    except (CorpusValidationError, OSError) as exc:
        raise KittiFixtureError(
            "KITTI source contract is unavailable or invalid"
        ) from exc
    if (
        set(contract) != SOURCE_CONTRACT_KEYS
        or contract.get("schema_version") != "1.0.0"
        or contract.get("fixture_id") != FIXTURE_ID
        or contract.get("status") != "ready"
        or contract.get("selection") != SELECTION
        or contract.get("license") != LICENSE
    ):
        raise KittiFixtureError("KITTI source contract is not release-ready")
    return contract


def _validate_archive(
    archive: Path, contract: dict[str, Any] | None
) -> str:
    if not _ordinary_file(archive):
        raise KittiFixtureError(
            "KITTI archive must be one ordinary non-reparse file"
        )
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise KittiFixtureError(
            "KITTI archive metadata differs from the official source contract"
        )
    actual = _sha256(archive)
    if contract is not None:
        source = contract.get("source_archive")
        if (
            not isinstance(source, dict)
            or source.get("filename") != ARCHIVE_FILENAME
            or source.get("url") != ARCHIVE_URL
            or source.get("bytes") != ARCHIVE_BYTES
            or source.get("etag") != ARCHIVE_ETAG
            or source.get("last_modified") != ARCHIVE_LAST_MODIFIED
            or actual != source.get("sha256")
        ):
            raise KittiFixtureError(
                "KITTI official archive metadata or SHA-256 mismatch"
            )
    return actual


def _frame_arrays(
    archive: zipfile.ZipFile,
    contract: dict[str, Any] | None,
    *,
    source_manifest: hashlib._Hash,
    transformed_rgb: hashlib._Hash,
) -> Iterable[Any]:
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ModuleNotFoundError as exc:
        raise KittiFixtureError(
            "The pinned release-corpus environment requires NumPy and Pillow"
        ) from exc

    if contract is not None and contract.get("selection") != SELECTION:
        raise KittiFixtureError("KITTI source selection contract drifted")

    for index in range(FIRST_FRAME, LAST_FRAME + 1):
        name = _member_name(index)
        try:
            info = archive.getinfo(name)
        except KeyError as exc:
            raise KittiFixtureError(
                f"KITTI source frame is missing: {name}"
            ) from exc
        mode = (int(info.external_attr) >> 16) & 0xFFFF
        if (
            info.is_dir()
            or stat.S_ISLNK(mode)
            or info.file_size <= 0
            or info.file_size > MAX_SOURCE_PNG_BYTES
            or info.compress_size <= 0
        ):
            raise KittiFixtureError(
                f"KITTI ZIP member is unsafe or oversized: {name}"
            )
        payload = archive.read(info)
        if len(payload) != info.file_size:
            raise KittiFixtureError(
                f"KITTI ZIP member is incomplete: {name}"
            )
        payload_sha = hashlib.sha256(payload).hexdigest()
        source_manifest.update(name.encode("ascii"))
        source_manifest.update(b"\0")
        source_manifest.update(payload_sha.encode("ascii"))
        source_manifest.update(b"\n")

        try:
            with Image.open(BytesIO(payload)) as source:
                source.load()
                if source.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    raise KittiFixtureError(
                        f"KITTI frame dimensions drifted: {name}"
                    )
                rgb = ImageOps.fit(
                    source.convert("RGB"),
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                array = np.asarray(rgb, dtype=np.uint8)
        except KittiFixtureError:
            raise
        except Exception as exc:
            raise KittiFixtureError(
                f"KITTI PNG failed strict decoding: {name}"
            ) from exc
        array = np.ascontiguousarray(array)
        transformed_rgb.update(array.tobytes(order="C"))
        yield array


def acquire(
    archive: Path,
    output: Path,
    *,
    measure: bool = False,
) -> dict[str, Any]:
    contract = None if measure else _require_source_contract()
    archive = Path(os.path.abspath(archive))
    archive_sha = _validate_archive(archive, contract)
    destination = _empty_plain_directory(output)
    video = destination / f"{FIXTURE_ID}.mp4"
    source_manifest = hashlib.sha256()
    transformed_rgb = hashlib.sha256()

    try:
        with zipfile.ZipFile(archive, "r", allowZip64=True) as source:
            _encode_video(
                video,
                frames=_frame_arrays(
                    source,
                    contract,
                    source_manifest=source_manifest,
                    transformed_rgb=transformed_rgb,
                ),
                frame_count=FRAME_COUNT,
                time_base=Fraction(1, OUTPUT_RATE),
                pts_values=range(FRAME_COUNT),
                color=COLOR_CASES["bt709-limited"],
                rate=OUTPUT_RATE,
                width=OUTPUT_WIDTH,
                height=OUTPUT_HEIGHT,
            )
    except (KittiFixtureError, zipfile.BadZipFile):
        raise
    except Exception as exc:
        raise KittiFixtureError(
            "KITTI fixture encoding failed closed"
        ) from exc

    source_manifest_sha = source_manifest.hexdigest()
    transformed_rgb_sha = transformed_rgb.hexdigest()
    if contract is not None:
        evidence = contract.get("observed_evidence")
        if (
            not isinstance(evidence, dict)
            or source_manifest_sha
            != evidence.get("ordered_member_sha256_manifest")
            or transformed_rgb_sha
            != evidence.get("transformed_rgb_sha256")
        ):
            raise KittiFixtureError(
                "KITTI selected source pixels differ from recorded evidence"
            )

    record = _record(
        video,
        fixture_id=FIXTURE_ID,
        kind="real-windowed-official-benchmark",
        extra={
            "official_dataset": (
                "KITTI Visual Odometry / SLAM Evaluation 2012"
            ),
            "sequence": SEQUENCE,
            "camera": CAMERA,
            "source_frame_range": [FIRST_FRAME, LAST_FRAME],
            "source_archive_sha256": archive_sha,
            "source_files": FRAME_COUNT,
            "source_manifest_sha256": source_manifest_sha,
            "transformed_rgb_sha256": transformed_rgb_sha,
            "license_spdx": "CC-BY-NC-SA-3.0",
        },
    )
    if contract is not None:
        expected_output = contract.get("expected_output")
        if not isinstance(expected_output, dict):
            raise KittiFixtureError(
                "KITTI expected output contract is malformed"
            )
        if record != expected_output:
            drifted = sorted(
                key
                for key in set(record) | set(expected_output)
                if record.get(key) != expected_output.get(key)
            )
            raise KittiFixtureError(
                "KITTI generated fixture drifted: "
                + ", ".join(drifted)
            )

    document = {
        "schema_version": "1.0.0",
        "status": (
            "measurement-only" if measure else "release-evidence"
        ),
        "fixture_id": FIXTURE_ID,
        "source_contract": (
            None
            if measure
            else {
                "path": SOURCE_CONTRACT.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    SOURCE_CONTRACT.read_bytes()
                ).hexdigest(),
            }
        ),
        "dataset": {
            "name": "KITTI Visual Odometry / SLAM Evaluation 2012",
            "official_page": DATASET_URL,
            "archive_url": ARCHIVE_URL,
        },
        "selection": SELECTION,
        "license": LICENSE if measure else contract["license"],
        "source_evidence": {
            "archive_filename": ARCHIVE_FILENAME,
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": archive_sha,
            "ordered_member_sha256_manifest": source_manifest_sha,
            "transformed_rgb_sha256": transformed_rgb_sha,
        },
        "output": record,
        "personal_capture_sources": False,
        "checked_in_media": False,
    }
    manifest_path = destination / (
        "benchmark-fixture-measurement.json"
        if measure
        else "benchmark-fixture-manifest.json"
    )
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive",
        type=Path,
        help="Official KITTI data_odometry_gray.zip archive",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="New or empty scratch output directory",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "Bootstrap observed checksums without claiming release evidence; "
            "the measured contract must be reviewed and checked in before a "
            "normal acquisition run"
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        result = acquire(
            arguments.archive,
            arguments.output,
            measure=arguments.measure,
        )
    except KittiFixtureError as exc:
        print(f"LINGBOT_MAP_KITTI_FIXTURE_ERROR={exc}", file=sys.stderr)
        return 2
    print(
        "LINGBOT_MAP_KITTI_FIXTURE="
        + json.dumps(
            {
                "fixture_id": result["fixture_id"],
                "frame_count": result["output"]["frame_count"],
                "sha256": result["output"]["sha256"],
                "status": result["status"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
