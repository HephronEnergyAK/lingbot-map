# LingBot Map release corpus 0.1.0

This directory is the versioned index and oracle set for non-personal release
fixtures. It intentionally contains no generated MP4/MOV files and no private
Capture Sources. Media is written only to a caller-supplied empty scratch
directory, verified against `generated-goldens.json`, and may then be deleted.

## Ordinary and milestone execution

Use the pinned Worker Runtime:

```powershell
python scripts/build_release_corpus.py C:\tmp\lingbot-map-release-corpus
python scripts/validate_release_corpus.py --generated-manifest C:\tmp\lingbot-map-release-corpus\generated-manifest.json
```

The ordinary build creates the five exact frame-count boundaries, VFR, four
accepted color descriptions, and the licensed real courthouse streaming
capture. The 25,000-frame fixture is deliberately reserved for a release
candidate or milestone:

```powershell
python scripts/build_release_corpus.py C:\tmp\lingbot-map-release-stress --include-stress
```

The output directory must be empty. The generator records the exact Python,
PyAV, NumPy, and FFmpeg library versions so a toolchain drift cannot silently
replace a golden.

## Layered oracles

`exact-oracles.json` is bit-exact for deterministic non-neural boundaries:
decoder display pixels and metadata, canonical preprocessing, automatic
pipeline choice, coordinate conversion, Point Reducer behavior, versioned
schema hashes, safe paths, and atomic Result handling.

`neural-oracles.json` explicitly forbids a cross-GPU output checksum. The
neural validator instead requires exact dtypes and aligned shapes, finite
outputs, rigid right-handed camera transforms, valid intrinsics, strictly
ordered source timestamps, complete provenance, and fixture-specific metric
ranges calibrated on the qualified RTX 5090 reference system. The calibration
uses a fixed `1e-6` orthonormal/determinant tolerance for rotations derived
from float32 pose output; the measured maximum reference error is `4.23e-7`.
The calibration
tool writes raw arrays and a separate hardware/metric report:

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-<physical-uuid>"
python scripts\calibrate_neural_oracle.py <fixture.mp4> <empty-output> `
  --report <new-report.json> --fixture-id <id> `
  --source-sha256 <sha256> --model <model.pt> `
  --model-sha256 <sha256> --runtime-id <runtime-id>
```

The committed ranges are rounded outward from one-half to twice the reference
metrics. This deliberately detects collapsed or explosive output rather than
requiring an invalid cross-architecture checksum.

The Blender 5.2 visual driver builds all eight display-oriented source
fixtures and measures animated camera, Source Background, and Model Coverage
alignment. Its success marker is valid only when the visible GUI run reports a
maximum error no greater than one display pixel.

Run it through the Windows wrapper so a temporary Blender profile disables the
fresh-install splash without changing the user's preferences:

```powershell
& tests\run_blender_source_view_visual.ps1 `
  -BlenderPath C:\path\to\blender.exe `
  -WorkerPython C:\path\to\pinned-worker-python.exe
```

## Current external release gates

The repository contains a redistributable real 286-frame streaming source
under the root Apache-2.0 license. It does not yet contain a real capture above
3,000 frames with explicit redistribution rights. The upstream demo
`indoor_travel.MP4` is recorded only as an unapproved candidate because its
dataset exposes no explicit license; it must not be downloaded into CI,
redistributed, or treated as release evidence.

Until a licensed long capture is supplied and the complete suite passes on an
Ada consumer GPU with at least 16 GB, structural validation succeeds but
`--release` validation fails with those exact blockers.
