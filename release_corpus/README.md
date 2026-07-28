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

## Official KITTI windowed fixture

The real Windowed fixture uses every `image_0` frame from 000000 through
003000 of sequence 00 in the official KITTI Visual Odometry / SLAM Evaluation
2012 grayscale archive. The 3,001-frame selection is a new Continuous Take:
frames remain in source order at 10 fps, with no temporal sampling. Only a
deterministic spatial fit to 518 by 158 pixels is applied before deterministic
CRF-0 H.264 encoding.

Download `data_odometry_gray.zip` from the archive URL recorded in
`kitti-odometry-00-source.json`, keep it outside the repository, and run:

```powershell
python scripts/acquire_kitti_release_fixture.py `
  C:\path\to\data_odometry_gray.zip `
  C:\tmp\lingbot-map-kitti-windowed
python scripts/validate_release_corpus.py `
  --benchmark-fixture-manifest `
  C:\tmp\lingbot-map-kitti-windowed\benchmark-fixture-manifest.json
```

The source archive, derived MP4, and acquisition manifest stay in
caller-supplied release-suite scratch storage. They are not included in the
Extension, Worker, repository, or ordinary CI artifacts. KITTI is separately
licensed under `CC-BY-NC-SA-3.0`: attribution, non-commercial use, and
ShareAlike 3.0 for derivatives are required. See
`licenses/KITTI-CC-BY-NC-SA-3.0.md` for the captured evidence and attribution.

The qualified RTX 5090 Draft calibration processed all 3,001 frames through
the real Windowed pipeline in 62 overlap boundaries. Its fixed v1 heuristic
reported 61 non-blocking Quality Warnings, which are preserved in the
calibration evidence and are not converted into a rejection gate. Every
boundary still supplied valid overlap data and a finite legal similarity
transform; the warning thresholds remain explicitly uncalibrated under
ADR 0028.

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

## Current release gates

The corpus has two licensed real sources: the repository Apache-2.0
286-frame courthouse sequence for Streaming and the externally stored
CC-BY-NC-SA-3.0 KITTI 3,001-frame sequence for Windowed. The upstream demo
`indoor_travel.MP4` remains only an unapproved candidate because its dataset
exposes no explicit license; it must not be downloaded into CI, redistributed,
or treated as release evidence.

The qualified RTX 5090 Blackwell suite is sufficient for a disclosed `0.x`
engineering Release Package, so `validate_release_corpus.py --release` does
not require Ada evidence. Such a release is Blackwell-qualified only and
cannot claim general NVIDIA, Ada, `1.0.0`, or official-platform readiness.

The complete native suite on an Ada consumer GPU with at least 16 GB remains a
`1.0.0` and official-platform gate, including the calibrated neural ranges and
the behavior when higher Profiles do not qualify.
`validate_release_corpus.py --stable-release` continues to fail on that exact
gate until real Ada evidence exists.
