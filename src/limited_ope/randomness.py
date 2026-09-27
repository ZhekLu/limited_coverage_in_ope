"""Semantic seed derivation independent of grid order or process scheduling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class SeedManager:
    """Derive reproducible streams using SHA-256 and NumPy SeedSequence.

    :param master_seed: Root integer recorded in the study manifest.
    """

    master_seed: int

    def seed(self, *labels: str | int | float) -> int:
        """Return a stable 128-bit seed for the supplied semantic labels.

        :param labels: Scope, cell coordinates, repetition, and stream name.
        :returns: A seed unaffected by iteration order or Python hash randomization.
        """
        encoded = json.dumps(labels, separators=(",", ":"), allow_nan=False).encode()
        words = np.frombuffer(hashlib.sha256(encoded).digest(), dtype="<u4")
        sequence = np.random.SeedSequence([self.master_seed, *map(int, words)])
        state = sequence.generate_state(4).astype("<u4")
        return int.from_bytes(state.tobytes(), byteorder="little")

    def generator(self, *labels: str | int | float) -> Generator:
        """Create an explicit PCG64 generator for a semantic stream."""
        return np.random.Generator(np.random.PCG64(self.seed(*labels)))
