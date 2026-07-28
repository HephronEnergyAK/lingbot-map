"""Explicit online probe for immutable Model Catalog sources (no weight download)."""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    catalog = json.loads((ROOT / "model-catalog.json").read_text(encoding="utf-8"))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    for model in catalog["models"]:
        artifact = model["artifact"]
        head = urllib.request.Request(artifact["url"], method="HEAD")
        with opener.open(head, timeout=60) as response:
            assert response.status == 200, response.status
            assert response.geturl().startswith("https://"), response.geturl()
            assert int(response.headers["Content-Length"]) == artifact["length"]
            assert response.headers.get("Accept-Ranges", "").lower() == "bytes"
            validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
            assert validator, response.headers
        request = urllib.request.Request(
            artifact["url"], headers={"Range": "bytes=0-0", "If-Range": validator}
        )
        with opener.open(request, timeout=60) as response:
            assert response.status == 206, response.status
            assert response.headers["Content-Range"] == f"bytes 0-0/{artifact['length']}"
            assert (response.headers.get("ETag") or response.headers.get("Last-Modified")) == validator
            assert len(response.read(2)) == 1
        results.append(
            {
                "model_id": model["id"],
                "revision": artifact["source_revision"],
                "length": artifact["length"],
                "range_status": 206,
                "range_bytes": 1,
                "validator": validator,
            }
        )
    print("LINGBOT_MAP_MODEL_SOURCE_PROBE=" + json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
