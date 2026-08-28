"""
mulberry32 PRNG, bit-identical to the JavaScript implementation in
web/js/engine.js.

The production Python engine uses numpy's PCG64. This exists purely so a run can
be reproduced exactly in both languages, which is what lets
tools/verify_web_engine.py assert that the browser port and the Python engine
produce the same spike trains rather than merely similar ones.
"""
from __future__ import annotations

import numpy as np

MASK = 0xFFFFFFFF


class Mulberry32:
    """Drop-in for numpy Generator's `.random(size)`, matching the JS version."""

    def __init__(self, seed: int = 0):
        self.a = int(seed) & MASK

    def _next_u32(self) -> int:
        self.a = (self.a + 0x6D2B79F5) & MASK
        a = self.a
        t = ((a ^ (a >> 15)) * (1 | a)) & MASK
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & MASK)) & MASK) ^ t) & MASK
        return (t ^ (t >> 14)) & MASK

    def random(self, size=None):
        if size is None:
            return self._next_u32() / 4294967296.0
        n = int(size)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._next_u32() / 4294967296.0
        return out
