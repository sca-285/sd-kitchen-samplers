# SPDX-License-Identifier: AGPL-3.0-or-later
# Compact rewrite of exponential integrators from ClownsharkBatwing / RES4LYF.
# Upstream may add terms beyond AGPL; see NOTICE.md and the RES4LYF LICENSE.

"""RES4LYF sampling loop (multistep, Adams-Bashforth, named RK tableaux). Work dtype is float64 on CPU/CUDA; callbacks stay float32."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from tqdm.auto import trange

try:
    from .exponential import (
        HistoryState, MultistepState, abnorsett_step, exp_rk_tableau_step, multistep_step,
    )
    from .tableaux import MULTISTEP_TABLEAUX, TABLEAUX
except ImportError:  # pragma: no cover - standalone load
    from exponential import (
        HistoryState, MultistepState, abnorsett_step, exp_rk_tableau_step, multistep_step,
    )
    from tableaux import MULTISTEP_TABLEAUX, TABLEAUX


MULTISTEP_ORDER = {"res_2m": 2, "res_3m": 3}


def method_names():
    """Every rk_type this loop accepts."""
    return [*MULTISTEP_ORDER, *TABLEAUX, *MULTISTEP_TABLEAUX]


def _work_dtype(device: torch.device) -> torch.dtype:
    """float64 on CPU/CUDA; float32 on MPS and similar."""
    return torch.float64 if device.type in ("cpu", "cuda") else torch.float32


def _randn_like(x):
    """Seeded noise: the WebUI's TorchHijack, so batch images match their seeds."""
    from lib_kitchen import host
    return host.randn_like(x)


def _ancestral_step(sigma_from, sigma_to, eta):
    """k-diffusion's get_ancestral_step: (sigma_down, sigma_up)."""
    if not eta:
        return sigma_to, sigma_to.new_zeros(())
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5,
    )
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_rk_beta(
    model,
    x: Tensor,
    sigmas: Tensor,
    sigmas_override: Optional[Tensor] = None,   # kept for positional compatibility
    extra_args: Optional[dict] = None,
    callback: Optional[Callable] = None,
    disable: bool = False,
    rk_type: str = "res_2m",
    eta: float = 0.0,
    s_noise: float = 1.0,
):
    extra_args = extra_args or {}
    out_dtype = x.dtype
    work = _work_dtype(x.device)
    x = x.to(work)
    sigmas = sigmas.to(work)
    s_in = x.new_ones([x.shape[0]])

    if rk_type in MULTISTEP_ORDER:
        scheme, state = "multistep", MultistepState(MULTISTEP_ORDER[rk_type])
    elif rk_type in MULTISTEP_TABLEAUX:
        depth = len(MULTISTEP_TABLEAUX[rk_type](0.5)[0])
        scheme, state = "abnorsett", HistoryState(depth)
    elif rk_type in TABLEAUX:
        scheme, state = "tableau", None
    else:
        raise ValueError(f"unknown RES4LYF method {rk_type!r}; known: {', '.join(method_names())}")

    def evaluate(x_stage, sigma_stage):
        """The model's denoised output, called in float32 and returned in `work`."""
        sigma_stage = torch.as_tensor(sigma_stage, dtype=work, device=x.device)
        denoised = model(x_stage.to(torch.float32), (sigma_stage * s_in).to(torch.float32),
                         **extra_args)
        return denoised.to(work)

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised_0 = evaluate(x, sigma)

        if callback is not None:
            s = sigma.to(out_dtype)
            callback({"x": x.to(out_dtype), "i": i, "sigma": s, "sigma_hat": s,
                      "denoised": denoised_0.to(out_dtype)})

        # Integrating to sigma = 0 is exactly x = denoised, as in k-diffusion.
        # Evaluating a final stage at sigma ~ 0 instead used to blow the last
        # step up by ~1e3.
        if sigma_next <= 0:
            x = denoised_0
            break

        # Ancestral split: deterministic step to sigma_down, then sigma_up of
        # fresh noise, so eta trades determinism for variety at a fixed noise level.
        sigma_down, sigma_up = _ancestral_step(sigma, sigma_next, eta)

        if scheme == "multistep":
            x_next = multistep_step(x, denoised_0, sigma, sigma_down, state)
        elif scheme == "abnorsett":
            x_next = abnorsett_step(x, denoised_0, sigma, sigma_down, rk_type, state, evaluate)
        else:
            x_next = exp_rk_tableau_step(x, denoised_0, sigma, sigma_down, rk_type, evaluate)

        if sigma_up > 0:
            x_next = x_next + _randn_like(x) * (s_noise * sigma_up)
        x = x_next

    return x.to(out_dtype)
