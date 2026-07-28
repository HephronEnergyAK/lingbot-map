"""Blender 5.2 smoke for redacted export and clipboard-only copy."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import bpy


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    with tempfile.TemporaryDirectory(
        prefix="lingbot-map-blender-diagnostics18-"
    ) as temporary:
        workspace = Path(temporary)
        blend = workspace / "target.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        project = workspace / "target.lingbot-map"
        diagnostic_name = (
            "job-0123456789abcdef0123456789abcdef"
            "--reconstruction-failed"
        )
        diagnostic = project / "diagnostics" / diagnostic_name
        diagnostic.mkdir(parents=True, exist_ok=True)
        (diagnostic / "diagnostic.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "error_code": "pipeline.reconstruction.failed",
                    "category": "pipeline",
                    "state": "failed",
                    "phase": "inference",
                    "job_id": (
                        "job-0123456789abcdef0123456789abcdef"
                    ),
                    "username": "PrivateBlenderUser",
                    "machine_name": "PRIVATE-BLENDER-HOST",
                    "project_root": r"D:\Private Blender Project",
                    "source": {
                        "absolute_path": (
                            r"C:\Users\PrivateBlenderUser\secret.mov"
                        ),
                        "sha256": "d" * 64,
                    },
                    "environment": {
                        "PRIVATE_TOKEN": "do-not-copy-this"
                    },
                    "gpu_uuid": (
                        "GPU-12345678-1234-1234-1234-123456789abc"
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (diagnostic / "human.log").write_text(
            (
                "PrivateBlenderUser PRIVATE-BLENDER-HOST "
                r"C:\Users\PrivateBlenderUser\secret.mov "
                "do-not-copy-this "
                + "d" * 64
            ),
            encoding="utf-8",
        )
        (diagnostic / "source.mov").write_bytes(b"source media")
        (diagnostic / "points.npy").write_bytes(b"point array")

        copied = bpy.ops.lingbot_map.copy_diagnostic_report(
            diagnostic_name=diagnostic_name
        )
        assert copied == {"FINISHED"}, copied
        clipboard = bpy.context.window_manager.clipboard
        clipboard_backend_available = bool(clipboard)
        if clipboard_backend_available:
            assert "pipeline.reconstruction.failed" in clipboard
            for secret in (
                "PrivateBlenderUser",
                "PRIVATE-BLENDER-HOST",
                "Private Blender Project",
                "secret.mov",
                "do-not-copy-this",
                "d" * 64,
                "GPU-12345678-1234-1234-1234-123456789abc",
            ):
                assert secret not in clipboard, secret

        redacted_zip = workspace / "redacted.zip"
        exported = bpy.ops.lingbot_map.export_diagnostic_report(
            diagnostic_name=diagnostic_name,
            filepath=str(redacted_zip),
            include_sensitive_identity=False,
        )
        assert exported == {"FINISHED"}, exported
        with zipfile.ZipFile(redacted_zip) as archive:
            redacted = b"\n".join(
                archive.read(name) for name in archive.namelist()
            )
            manifest = json.loads(
                archive.read("report-manifest.json")
            )
        assert manifest["identity_mode"] == "redacted"
        assert b"source media" not in redacted
        assert b"point array" not in redacted
        assert b"PrivateBlenderUser" not in redacted

        unredacted_zip = workspace / "unredacted.zip"
        exported = bpy.ops.lingbot_map.export_diagnostic_report(
            diagnostic_name=diagnostic_name,
            filepath=str(unredacted_zip),
            include_sensitive_identity=True,
        )
        assert exported == {"FINISHED"}, exported
        with zipfile.ZipFile(unredacted_zip) as archive:
            unredacted = b"\n".join(
                archive.read(name) for name in archive.namelist()
            )
            unredacted_manifest = json.loads(
                archive.read("report-manifest.json")
            )
        assert unredacted_manifest["identity_mode"] == "unredacted"
        assert b"PrivateBlenderUser" in unredacted
        assert b"source media" not in unredacted
        assert b"point array" not in unredacted

        print(
            "LINGBOT_MAP_BLENDER_DIAGNOSTICS="
            + json.dumps(
                {
                    "blender_version": bpy.app.version_string,
                    "copy_operator_finished": True,
                    "clipboard_backend_available": (
                        clipboard_backend_available
                    ),
                    "redacted_export": True,
                    "explicit_unredacted_export": True,
                    "allowlist_exclusions": True,
                    "clipboard_bytes": len(
                        clipboard.encode("utf-8")
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )


if __name__ == "__main__":
    main()
