# SPDX-License-Identifier: AGPL-3.0-or-later
# Compact rewrite of exponential integrators from ClownsharkBatwing / RES4LYF.
# Upstream may add terms beyond AGPL; see NOTICE.md and the RES4LYF LICENSE.

"""Exponential integrators for dx/dsigma = (x - D)/sigma, written in t = -log(sigma).

Families: res_2m/3m (multistep), abnorsett_* (Adams-Bashforth), res_2s…res_6s (named RK).
"""

from __future__ import annotations

import math

try:
    from .tableaux import build
except ImportError:  # pragma: no cover - standalone load
    from tableaux import build


def step_size(sigma, sigma_down):
    """h = log(sigma / sigma_down), the step in t = -log(sigma)."""
    return math.log(float(sigma) / float(sigma_down))


def _combine(x_0, h, weights, eps):
    """x_0 + h * sum_j weights[j] * eps[j], skipping zero weights."""
    acc = None
    for w, e in zip(weights, eps):
        if w == 0:
            continue
        term = w * e
        acc = term if acc is None else acc + term
    return x_0 if acc is None else x_0 + h * acc


# ----------------------------------------------------------------- multistep


class MultistepState:
    """The previous denoised outputs and step sizes res_2m/res_3m extrapolate from."""

    __slots__ = ("d1", "d2", "h1", "h2", "order")

    def __init__(self, order):
        self.order = order
        self.d1 = self.d2 = None
        self.h1 = self.h2 = None

    def push(self, denoised, h):
        self.d1, self.d2 = denoised, self.d1
        self.h1, self.h2 = h, self.h1


def multistep_step(x, denoised, sigma, sigma_down, state):
    """One DPM-Solver++(2M)/(3M) update, alpha_t = 1.

    The first step has no history and is plain exponential Euler, as in every
    published implementation. An ETDRK start-up was measured against it on a
    Gaussian-mixture testbed and lost on both cost and error, because the first
    step sits at high sigma where exponential Euler is very nearly exact.
    """
    h = step_size(sigma, sigma_down)
    e = math.exp(-h)

    # Exponential Euler base step (this alone is DDIM).
    x = e * x + (1.0 - e) * denoised

    phi_2 = math.expm1(-h) / h + 1.0
    if state.order >= 3 and state.h2 is not None:
        r0 = state.h1 / h
        r1 = state.h2 / h
        d1_0 = (denoised - state.d1) / r0
        d1_1 = (state.d1 - state.d2) / r1
        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
        d2 = (d1_0 - d1_1) / (r0 + r1)
        # 2 * (phi_2/h - 1/2), not (phi_2/h - 1/2): the factor pairs with the
        # 1/(r0+r1) normalisation of d2. Measured: 3.01 with it, 1.92 without.
        phi_3 = 2.0 * (phi_2 / h - 0.5)
        x = x + phi_2 * d1 - phi_3 * d2
    elif state.h1 is not None:
        x = x + phi_2 * (denoised - state.d1) / (state.h1 / h)

    state.push(denoised, h)
    return x


# --------------------------------------------------- named tableaux (RES4LYF)


def exp_rk_tableau_step(x_0, denoised_0, sigma, sigma_down, name, evaluate):
    """One step of a named exponential Runge-Kutta method.

    Upstream RES4LYF's convention, everything against the step's start:

        x_i    = x_0 + h * sum_j a[i][j] * (D_j - x_0)   at sigma * exp(-h c_i)
        x_next = x_0 + h * sum_j b[0][j] * (D_j - x_0)

    `evaluate(x_stage, sigma_stage)` returns the denoised output at a stage;
    `denoised_0` is the one the caller already has at `sigma`.
    """
    h = step_size(sigma, sigma_down)
    ci, a, b = build(name, h)

    eps = [denoised_0 - x_0]
    for i in range(1, len(ci)):
        x_i = _combine(x_0, h, a[i], eps)
        eps.append(evaluate(x_i, float(sigma) * math.exp(-h * float(ci[i]))) - x_0)
    return _combine(x_0, h, b[0], eps)


# ------------------------------------------------ exponential Adams-Bashforth


class HistoryState:
    """The past DENOISED outputs an exponential Adams-Bashforth method uses.

    Denoised, not epsilons: every abnorsett weight row sums to phi_1(-h), which
    is only consistent if each weight multiplies D_k - x_0 with x_0 the CURRENT
    step's start. Storing per-step epsilons instead runs, makes an image, and
    converges at order 0.8.
    """

    __slots__ = ("depth", "denoised")

    def __init__(self, depth: int):
        self.depth = int(depth)
        self.denoised: list = []

    def push(self, denoised):
        self.denoised.insert(0, denoised)
        del self.denoised[self.depth:]


# Single-step method run while history is still short. Starting abnorsett_3m or
# _4m with exponential Euler caps the whole run at order 2.2; a start-up of
# matching order costs a few extra model calls on the first steps only.
STARTUP = {
    "abnorsett_2m": None,
    "abnorsett_3m": "res_3s",
    "abnorsett_4m": "res_4s_krogstad",
}


def abnorsett_step(x_0, denoised_0, sigma, sigma_down, name, state, evaluate):
    """One exponential Adams-Bashforth step; no extra model calls once warmed up.

    The weights assume a constant h, which a geometric (exponential) sigma
    schedule gives. On karras and friends the method still converges, below
    its nominal order.
    """
    h = step_size(sigma, sigma_down)
    state.push(denoised_0)

    if len(state.denoised) < state.depth:
        startup = STARTUP.get(name)
        if startup is not None:
            return exp_rk_tableau_step(x_0, denoised_0, sigma, sigma_down, startup, evaluate)
        e = math.exp(-h)
        return e * x_0 + (1.0 - e) * denoised_0

    _, _, b = build(name, h)
    return _combine(x_0, h, b[0], [d - x_0 for d in state.denoised])
