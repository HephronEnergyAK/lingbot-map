"""Independent Apache-2.0 LingBot Map Reconstruction Worker package."""

from __future__ import annotations


__version__ = "0.1.0"
WORKER_DISTRIBUTION = "lingbot-map-worker"


def identity() -> dict[str, str]:
    """Return the bounded identity used by Runtime post-sync validation."""

    return {
        "distribution": WORKER_DISTRIBUTION,
        "version": __version__,
    }
