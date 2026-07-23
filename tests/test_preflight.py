from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
WORKER_SRC = ROOT / "worker" / "src"
if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

try:
    import av
    import numpy as np
except ModuleNotFoundError:  # The Blender-side test environment does not install Worker wheels.
    av = None
    np = None

if av is not None and np is not None:
    from lingbot_map_worker import decoder


class _Disposition:
    def __init__(self, *, default=False, attached_pic=False):
        self.default = default
        self.attached_pic = attached_pic


class _Format:
    def __init__(self, name="yuv420p", *, alpha=False):
        self.name = name
        self.components = [SimpleNamespace(is_alpha=alpha)]


class _Frame:
    def __init__(
        self,
        index: int,
        *,
        transform=(1, 0, 0, 1),
        color=(1, 1, 1, 1),
        interlaced=False,
        alpha=False,
        sar=Fraction(1, 1),
        width=3,
        height=2,
        pts=None,
    ):
        self.pts = index if pts is None else pts
        self.time_base = Fraction(1, 10)
        self.width = width
        self.height = height
        self.interlaced_frame = interlaced
        self.is_corrupt = False
        self.sample_aspect_ratio = sar
        self.color_primaries, self.color_trc, self.colorspace, self.color_range = color
        self.format = _Format(alpha=alpha)
        a, b, c, d = transform
        values = (a * 65536, b * 65536, 0, c * 65536, d * 65536, 0, 0, 0, 1 << 30)
        self.side_data = {"DISPLAYMATRIX": struct.pack("=9i", *values)} if index == 0 else {}
        self._rgb = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)

    def to_ndarray(self, *, format):
        if format != "rgb24":
            raise AssertionError("canonical decode must request rgb24")
        return self._rgb.copy()


class _Stream:
    def __init__(self, index=0, *, default=False, attached_pic=False):
        self.index = index
        self.disposition = _Disposition(default=default, attached_pic=attached_pic)
        self.time_base = Fraction(1, 10)
        self.average_rate = Fraction(10, 1)
        self.sample_aspect_ratio = Fraction(1, 1)
        self.metadata = {}
        self.codec_context = SimpleNamespace(
            name="h264", thread_count=0, thread_type="", sample_aspect_ratio=Fraction(1, 1)
        )


class _Container:
    def __init__(self, frames, *, video_streams=None, audio_streams=()):
        self._frames = frames
        self._selected = None
        self.streams = SimpleNamespace(
            video=list(video_streams or [_Stream(default=True)]), audio=list(audio_streams)
        )
        self.format = SimpleNamespace(name="mov,mp4,m4a,3gp,3g2,mj2")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def decode(self, stream):
        self._selected = stream
        yield from self._frames


@unittest.skipUnless(av is not None and np is not None, "PyAV is installed only in the pinned Worker Runtime")
class DecoderAdapterTests(unittest.TestCase):
    def _preflight(self, frames, *, video_streams=None, audio_streams=(), cancel=lambda: False):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "capture.mp4"
            source.write_bytes(b"fixture media identity")
            container = _Container(
                frames, video_streams=video_streams, audio_streams=audio_streams
            )
            with mock.patch.object(decoder.av, "open", return_value=container):
                report = decoder.preflight_capture_source(source, cancel=cancel)
            return report, container

    def test_all_color_range_and_eight_transform_combinations_are_deterministic(self):
        colors = ((1, 1, 1, 1), (1, 1, 1, 2), (5, 6, 6, 1), (6, 5, 5, 2))
        expected_standard = ("bt709", "bt709", "bt601", "bt601")
        expected_range = ("limited", "full", "limited", "full")
        for transform in decoder.TRANSFORMS:
            for color_index, color in enumerate(colors):
                with self.subTest(transform=transform, color=color):
                    frames = [_Frame(index, transform=transform, color=color) for index in range(8)]
                    first, _ = self._preflight(frames)
                    second, _ = self._preflight(frames)
                    self.assertEqual(first.display_transform, decoder.TRANSFORMS[transform])
                    self.assertEqual(first.color.standard, expected_standard[color_index])
                    self.assertEqual(first.color.range, expected_range[color_index])
                    self.assertEqual(first.rgb_sha256, second.rgb_sha256)
                    self.assertEqual(len(first.timestamps_seconds), 8)

    def test_transform_pixels_have_exact_oracles_and_unsupported_matrices_fail(self):
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        expected = {
            (1, 0, 0, 1): rgb,
            (0, -1, 1, 0): np.rot90(rgb, 1),
            (-1, 0, 0, -1): np.rot90(rgb, 2),
            (0, 1, -1, 0): np.rot90(rgb, 3),
            (-1, 0, 0, 1): np.flip(rgb, 1),
            (1, 0, 0, -1): np.flip(rgb, 0),
            (0, 1, 1, 0): np.transpose(rgb, (1, 0, 2)),
            (0, -1, -1, 0): np.flip(np.transpose(rgb, (1, 0, 2)), (0, 1)),
        }
        for matrix, oracle in expected.items():
            np.testing.assert_array_equal(
                decoder.apply_display_transform(rgb, decoder.TRANSFORMS[matrix]), oracle
            )
        rejected = (
            (2 << 16, 0, 0, 0, 1 << 16, 0, 0, 0, 1 << 30),
            (1 << 16, 1 << 15, 0, 0, 1 << 16, 0, 0, 0, 1 << 30),
            (1 << 16, 0, 0, 0, 1 << 16, 0, 1, 0, 1 << 30),
        )
        for values in rejected:
            with self.assertRaises(decoder.DecoderError):
                decoder._normalized_matrix(values)

    def test_track_selection_requires_one_unique_default_and_never_decodes_audio(self):
        attached = _Stream(0, default=True, attached_pic=True)
        selected = _Stream(2, default=True)
        other = _Stream(1)
        audio = SimpleNamespace(
            index=3,
            codec_context=SimpleNamespace(
                name="aac", sample_rate=48000,
                layout=SimpleNamespace(nb_channels=2, channels=(1, 2)),
            ),
        )
        report, container = self._preflight(
            [_Frame(index) for index in range(8)],
            video_streams=[attached, other, selected],
            audio_streams=[audio],
        )
        self.assertIs(container._selected, selected)
        self.assertEqual(report.video_stream_index, 2)
        self.assertEqual(report.audio_streams[0]["codec"], "aac")
        ambiguous = [_Stream(0), _Stream(1)]
        with self.assertRaisesRegex(decoder.DecoderError, "unique default"):
            self._preflight([_Frame(index) for index in range(8)], video_streams=ambiguous)

    def test_every_rejected_frame_class_minimum_vfr_cancel_late_failure_and_replacement(self):
        cases = {
            "interlaced": [_Frame(index, interlaced=index == 7) for index in range(8)],
            "alpha": [_Frame(index, alpha=index == 7) for index in range(8)],
            "non-square": [_Frame(index, sar=Fraction(2, 1) if index == 7 else Fraction(1, 1)) for index in range(8)],
            "HDR": [_Frame(index, color=(9, 16, 9, 1)) for index in range(8)],
            "range": [_Frame(index, color=(1, 1, 1, 0)) for index in range(8)],
            "changes": [_Frame(index, width=4 if index == 7 else 3) for index in range(8)],
            "timestamp": [_Frame(index, pts=0 if index == 7 else index) for index in range(8)],
            "at least": [_Frame(index) for index in range(7)],
        }
        corrupt = [_Frame(index) for index in range(8)]
        corrupt[-1].is_corrupt = True
        cases["corrupt"] = corrupt
        for message, frames in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(decoder.DecoderError, message):
                self._preflight(frames)

        checks = iter((False, False, True))
        with self.assertRaises(decoder.PreflightCancelled):
            self._preflight([_Frame(index) for index in range(8)], cancel=lambda: next(checks, True))

        def late_frames():
            yield from [_Frame(index) for index in range(7)]
            raise av.error.InvalidDataError(0, "late fixture decode failure")

        with self.assertRaisesRegex(decoder.DecoderError, "Decode failed at frame 7"):
            self._preflight(late_frames())

        with mock.patch.object(
            decoder, "require_same_source",
            side_effect=decoder.DecoderError("Capture Source identity changed after preflight"),
        ):
            with self.assertRaisesRegex(decoder.DecoderError, "changed"):
                self._preflight([_Frame(index) for index in range(8)])

    def test_real_mp4_decodes_every_vfr_presentation_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "vfr.mp4"
            time_base = Fraction(1, 1000)
            pts_values = (0, 33, 67, 101, 151, 184, 250, 284)
            with av.open(str(source), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height = 16, 12
                stream.pix_fmt = "yuv420p"
                stream.time_base = time_base
                stream.codec_context.color_primaries = 1
                stream.codec_context.color_trc = 1
                stream.codec_context.colorspace = 1
                stream.codec_context.color_range = 1
                for index, pts in enumerate(pts_values):
                    array = np.full((12, 16, 3), index * 20, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    frame.pts, frame.time_base = pts, time_base
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(source), "r") as container:
                expected = tuple(float(frame.pts * frame.time_base) for frame in container.decode(container.streams.video[0]))
            report = decoder.preflight_capture_source(source)
            self.assertEqual(len(report.timestamps_seconds), 8)
            self.assertEqual(report.timestamps_seconds, expected)
            self.assertTrue(report.variable_frame_rate)
            self.assertEqual(report.color.standard, "bt709")
            self.assertEqual(report.color.range, "limited")


if __name__ == "__main__":
    unittest.main()
