# SPDX-License-Identifier: AGPL-3.0-or-later
# Compact rewrite of exponential integrators from ClownsharkBatwing / RES4LYF.
# Upstream may add terms beyond AGPL; see NOTICE.md and the RES4LYF LICENSE.

"""phi functions for exponential RK: series for |z| < 2, recurrence otherwise.

    phi_0(z) = e^z
    phi_{j+1}(z) = (phi_j(z) - 1/j!) / z ,   phi_j(0) = 1/j!
    phi_j(z) = sum_{k>=0} z^k / (k+j)!
"""

from __future__ import annotations

import math

SERIES_BELOW = 2.0
SERIES_TERMS = 40


def phi(j: int, z: float) -> float:
    """phi_j(z), j >= 0."""
    j = int(j)
    if j < 0:
        raise ValueError(f"phi order must be >= 0, got {j}")
    z = float(z)
    if j == 0:
        return math.exp(z)
    if z == 0.0:
        return 1.0 / math.factorial(j)

    if abs(z) < SERIES_BELOW:
        # sum_{k>=0} z^k / (k+j)!, accumulated as a running term so no
        # factorial larger than needed is ever formed.
        term = 1.0 / math.factorial(j)
        total = term
        for k in range(1, SERIES_TERMS):
            term *= z / (k + j)
            total += term
            if abs(term) <= abs(total) * 1e-18:
                break
        return total

    # phi_{k+1} = (phi_k - 1/k!) / z, up from phi_0 = e^z.
    value = math.exp(z)
    for k in range(j):
        value = (value - 1.0 / math.factorial(k)) / z
    return value


class Phi:
    """Upstream's tableau notation, verbatim.

    A tableau writes `phi(2, 3)` for "phi_2 evaluated at the third stage node",
    and `phi(2)` for the full step. Both are functions of -h, because the linear
    part of the diffusion ODE in t = -log(sigma) is exactly -x:

        phi(j)      = phi_j(-h)
        phi(j, i)   = phi_j(-h * c_i)      with i counted from 1

    A node at c_i = 0 makes every phi_j(0) a constant, and upstream returns a
    plain 0 there rather than 1/j!; that is not a simplification, it is what the
    tableaux are written against - those entries are always multiplied into a
    stage that contributes nothing - so this reproduces it exactly.
    """

    __slots__ = ("h", "c", "_cache")

    def __init__(self, h, c):
        self.h = float(h)
        self.c = list(c)
        self._cache: dict[tuple[int, int], float] = {}

    def __call__(self, j: int, i: int = -1):
        key = (j, i)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if i < 0:
            scale = 1.0
        else:
            scale = self.c[i - 1]
            if scale == 0:
                self._cache[key] = 0
                return 0
        value = phi(j, -self.h * scale)
        self._cache[key] = value
        return value


def calculate_gamma(c2, c3):
    """Upstream's helper, used by the free-parameter tableaux."""
    return (3 * (c3 ** 3) - 2 * c3) / (c2 * (2 - 3 * c2))


def gen_first_col_exp(a, b, c, phi_fn):
    """Fill in column 0 so every row sums to what consistency requires.

    A tableau is written with its first column blank because that column is
    forced: stage i must reproduce the exponential Euler step to its own node,
    so its row has to sum to c_i * phi_1(-c_i h), and the weight row has to sum
    to phi_1(-h). Upstream generates it rather than writing it out, and getting
    this wrong is invisible - the method stays stable and simply drops an order.
    """
    for i in range(len(c)):
        a[i][0] = c[i] * phi_fn(1, i + 1) - sum(a[i])
    for i in range(len(b)):
        b[i][0] = phi_fn(1) - sum(b[i])
    return a, b
