"""Pure, read-only Supported Host detection for the Blender Extension."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import struct
import sys
from typing import Mapping, Sequence


SUPPORTED_BLENDER_SERIES = (5, 2)
WINDOWS_11_MINIMUM_BUILD = 22000
WINDOWS_WORKSTATION_PRODUCT_TYPE = 1


@dataclass(frozen=True)
class HostFacts:
    """Read-only facts used to make one deterministic support decision."""

    operating_system: str
    operating_system_release: str
    native_architecture: str
    pointer_bits: int
    blender_version: tuple[int, int, int]
    windows_build: int | None = None
    windows_product_type: int | None = None
    compatibility_layer: str | None = None


@dataclass(frozen=True)
class HostDecision:
    """Fail-fast result consumed by every future mutating Setup action."""

    supported: bool
    code: str
    message: str
    facts: HostFacts


def collect_host_facts(
    blender_version: Sequence[int],
    *,
    environ: Mapping[str, str] | None = None,
) -> HostFacts:
    """Inspect the current process without touching storage or networking."""

    environment = os.environ if environ is None else environ
    system = platform.system()
    release = platform.release()
    native_architecture = (
        environment.get("PROCESSOR_ARCHITEW6432")
        or environment.get("PROCESSOR_ARCHITECTURE")
        or platform.machine()
    )
    compatibility_layer = _detect_compatibility_layer(environment)
    windows_build = None
    windows_product_type = None

    get_windows_version = getattr(sys, "getwindowsversion", None)
    if system == "Windows" and get_windows_version is not None:
        version = get_windows_version()
        windows_build = int(version.build)
        windows_product_type = int(version.product_type)

    normalized_version = tuple(int(part) for part in blender_version[:3])
    if len(normalized_version) != 3:
        raise ValueError("blender_version must contain major, minor, and patch")

    return HostFacts(
        operating_system=system,
        operating_system_release=release,
        native_architecture=native_architecture,
        pointer_bits=struct.calcsize("P") * 8,
        blender_version=normalized_version,
        windows_build=windows_build,
        windows_product_type=windows_product_type,
        compatibility_layer=compatibility_layer,
    )


def evaluate_supported_host(facts: HostFacts) -> HostDecision:
    """Apply the exact native-v1 Windows 11 x64 and Blender 5.2 gate."""

    if facts.operating_system != "Windows":
        return _unsupported(
            facts,
            "unsupported_os",
            "LingBot Map Reconstruction 0.1.0 supports only Windows 11 x64",
        )
    if facts.compatibility_layer:
        return _unsupported(
            facts,
            "unsupported_compatibility_layer",
            f"Compatibility layer {facts.compatibility_layer} is not supported",
        )

    architecture = facts.native_architecture.casefold().replace("-", "_")
    if architecture not in {"amd64", "x86_64"} or facts.pointer_bits != 64:
        return _unsupported(
            facts,
            "unsupported_architecture",
            "A native Windows x64 Blender process is required",
        )
    if facts.windows_product_type != WINDOWS_WORKSTATION_PRODUCT_TYPE:
        return _unsupported(
            facts,
            "unsupported_windows_edition",
            "Windows Server editions are not supported",
        )
    if (
        facts.operating_system_release != "11"
        or facts.windows_build is None
        or facts.windows_build < WINDOWS_11_MINIMUM_BUILD
    ):
        return _unsupported(
            facts,
            "unsupported_windows_version",
            "Windows 11 is required",
        )
    if facts.blender_version[:2] != SUPPORTED_BLENDER_SERIES:
        return _unsupported(
            facts,
            "unsupported_blender_version",
            "Blender 5.2 LTS is required",
        )

    return HostDecision(
        supported=True,
        code="supported",
        message="Supported Windows 11 x64 and Blender 5.2 LTS host",
        facts=facts,
    )


def probe_supported_host(blender_version: Sequence[int]) -> HostDecision:
    """Collect and evaluate current host facts without side effects."""

    return evaluate_supported_host(collect_host_facts(blender_version))


def _detect_compatibility_layer(environment: Mapping[str, str]) -> str | None:
    if "WINEPREFIX" in environment or "WINELOADERNOEXEC" in environment:
        return "Wine"
    return None


def _unsupported(facts: HostFacts, code: str, message: str) -> HostDecision:
    return HostDecision(supported=False, code=code, message=message, facts=facts)
