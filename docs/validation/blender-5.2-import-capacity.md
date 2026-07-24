# Blender 5.2 Import Capacity calibration

Issue: #14

Probe: `tests/blender_import_benchmark.py`

Host: Windows 11 x64

Blender: 5.2.0 LTS, build `fbe6228777e7`

Measurement: fresh background process for each point count; Windows
`K32GetProcessMemoryInfo(...).PrivateUsage` before and after the complete live
set of native PointCloud data, imported attributes, material, Geometry Nodes,
and NumPy bulk buffers.

| Points | Baseline private bytes | Final private bytes | Delta bytes |
| ---: | ---: | ---: | ---: |
| 0 | 256,180,224 | 263,327,744 | 7,147,520 |
| 100,000 | 256,151,552 | 276,881,408 | 20,729,856 |
| 1,000,000 | 257,204,224 | 344,604,672 | 87,400,448 |
| 5,000,000 | 257,052,672 | 661,458,944 | 404,406,272 |

Subtracting the zero-point fixed delta gives observed variable costs of 135.83
bytes/point at 100,000 points, 80.26 bytes/point at 1,000,000 points, and 79.45
bytes/point at 5,000,000 points. The pinned gate rounds this envelope upward to:

- fixed import cost: 256 MiB;
- variable import cost: 160 bytes per actual retained point;
- required headroom: 125% of the fixed-plus-variable estimate;
- additional physical-memory reserve: 4 GiB.

Runtime uses current available physical memory from `GlobalMemoryStatusEx`, not
pagefile capacity. Both manual and automatic import reject below the resulting
threshold. There is no override, partial Collection, or import-time downsample
path.
