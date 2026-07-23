"""Process-local, presentation-only Extension shell state."""

from __future__ import annotations

from .host import HostDecision


_host_decision: HostDecision | None = None


def set_host_decision(decision: HostDecision) -> None:
    global _host_decision
    _host_decision = decision


def get_host_decision() -> HostDecision | None:
    return _host_decision


def clear_host_decision() -> None:
    global _host_decision
    _host_decision = None
