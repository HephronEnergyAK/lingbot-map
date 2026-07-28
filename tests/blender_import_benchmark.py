"""Blender 5.2 Windows x64 import-capacity calibration probe.

Run each point count in a fresh Blender process. The probe includes native
PointCloud storage, all four imported attributes, and the largest simultaneous
Python/NumPy bulk buffers used by the importer. Reported PrivateUsage deltas are
inputs to a conservative rounded upper-envelope coefficient, not runtime gates.
"""

from __future__ import annotations

import ctypes
import json
import sys

import bpy
import numpy as np


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    )


def private_usage() -> int:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    query = kernel32.K32GetProcessMemoryInfo
    query.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        ctypes.c_ulong,
    )
    query.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not query(
        handle, ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PrivateUsage)


def main() -> None:
    assert bpy.app.version[:2] == (5, 2), bpy.app.version
    separator = sys.argv.index("--")
    point_count = int(sys.argv[separator + 1])
    baseline = private_usage()

    positions = np.zeros((point_count, 3), dtype=np.float32)
    colors = np.zeros((point_count, 3), dtype=np.uint8)
    confidence = np.ones(point_count, dtype=np.float32)
    radius = np.ones(point_count, dtype=np.float32)
    source_frame = np.zeros(point_count, dtype=np.uint32)
    rgba = np.empty((point_count, 4), dtype=np.float32)
    rgba[:, :3] = colors.astype(np.float32) / 255.0
    rgba[:, 3] = 1.0

    cloud = bpy.data.pointclouds.new("Capacity Benchmark")
    cloud.resize(point_count)
    cloud.points.foreach_set("co", positions.reshape(-1))
    color = cloud.attributes.new("color", "BYTE_COLOR", "POINT")
    color.data.foreach_set("color_srgb", rgba.reshape(-1))
    attribute = cloud.attributes.new("confidence", "FLOAT", "POINT")
    attribute.data.foreach_set("value", confidence)
    radius_attribute = cloud.attributes.get("radius")
    if radius_attribute is None:
        radius_attribute = cloud.attributes.new("radius", "FLOAT", "POINT")
    radius_attribute.data.foreach_set("value", radius)
    attribute = cloud.attributes.new("source_frame", "INT", "POINT")
    attribute.data.foreach_set("value", source_frame.astype(np.int32))
    object_ = bpy.data.objects.new("Capacity Benchmark", cloud)
    bpy.context.scene.collection.objects.link(object_)

    material = bpy.data.materials.new("Capacity Benchmark Material")
    material.use_nodes = True
    cloud.materials.append(material)
    geometry = bpy.data.node_groups.new(
        "Capacity Benchmark Geometry", "GeometryNodeTree"
    )
    geometry.interface.new_socket(
        name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    geometry.interface.new_socket(
        name="Radius Scale", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    geometry.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    group_input = geometry.nodes.new("NodeGroupInput")
    group_output = geometry.nodes.new("NodeGroupOutput")
    geometry.links.new(
        group_input.outputs["Geometry"], group_output.inputs["Geometry"]
    )
    modifier = object_.modifiers.new("Capacity Benchmark", "NODES")
    modifier.node_group = geometry
    bpy.context.view_layer.update()

    final = private_usage()
    result = {
        "blender": bpy.app.version_string,
        "platform": "windows-x64",
        "points": point_count,
        "baseline_private_bytes": baseline,
        "peak_live_private_bytes": final,
        "delta_bytes": max(0, final - baseline),
        "delta_bytes_per_point": (
            max(0, final - baseline) / point_count if point_count else 0.0
        ),
    }
    print("LINGBOT_MAP_IMPORT_BENCHMARK=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
