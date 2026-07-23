"""Generate one tiny deterministic VFR MP4 with pinned PyAV for integration tests."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import av
import numpy as np


def main() -> int:
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    time_base = Fraction(1, 1000)
    pts_values = (0, 33, 67, 101, 151, 184, 250, 284)
    with av.open(str(destination), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 1
        stream.codec_context.colorspace = 1
        stream.codec_context.color_range = 1
        for index, pts in enumerate(pts_values):
            x = np.arange(32, dtype=np.uint8)[None, :]
            y = np.arange(24, dtype=np.uint8)[:, None]
            rgb = np.empty((24, 32, 3), dtype=np.uint8)
            rgb[:, :, 0] = (x.astype(np.uint16) + index * 17).astype(np.uint8)
            rgb[:, :, 1] = (y.astype(np.uint16) + index * 29).astype(np.uint8)
            rgb[:, :, 2] = (
                x.astype(np.uint16) + y.astype(np.uint16) + index * 11
            ).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
