# SPDX-License-Identifier: AGPL-3.0-or-later
# Compact rewrite of exponential integrators from ClownsharkBatwing / RES4LYF.
# Upstream may add terms beyond AGPL; see NOTICE.md and the RES4LYF LICENSE.

"""RES4LYF dropdown entries: noisy (`eta`, `s_noise`) and ODE (`eta` pinned to 0). Duplicate algorithms are aliases in LEGACY."""

from __future__ import annotations

import torch

from lib_kitchen.res4lyf.rk_sampler_beta import sample_rk_beta


# (method, model calls per step). STARTUP_CALLS are extra first-step evaluations.
STARTUP_CALLS = {"abnorsett_3m": 4, "abnorsett_4m": 9}

METHODS = [
    ("res_2m", 1),
    ("res_3m", 1),
    ("abnorsett_2m", 1),
    ("abnorsett_3m", 1),
    ("abnorsett_4m", 1),
    ("res_2s", 2),
    ("res_2s_stable", 2),
    ("res_3s", 3),
    ("res_3s_non-monotonic", 3),
    ("res_3s_alt", 3),
    ("res_3s_cox_matthews", 3),
    ("res_3s_lie", 3),
    ("res_3s_sunstar", 3),
    ("res_3s_strehmel_weiner", 3),
    ("res_4s_krogstad", 4),
    ("res_4s_krogstad_alt", 4),
    ("res_4s_strehmel_weiner", 4),
    ("res_4s_strehmel_weiner_alt", 4),
    ("res_4s_cox_matthews", 4),
    ("res_4s_cfree4", 4),
    ("res_4s_friedli", 4),
    ("res_4s_minchev", 4),
    ("res_4s_munthe-kaas", 4),
    ("res_5s", 5),
    ("res_5s_hochbruck-ostermann", 5),
    ("res_6s", 6),
]

# Same algorithm as the entry they now point at. Noisy res_2m is not Res Multistep Ancestral.
LEGACY = {
    "res_2s_stable": (["RES4LYF res_2s_rkmk2e", "res_2s_rkmk2e"],
                      ["RES4LYF res_2s_rkmk2e ODE", "res_2s_rkmk2e_ode",
                       "Exp Heun 2 x0", "k_exp_heun_2_x0"]),
    "res_4s_cox_matthews": (["RES4LYF res_4s", "res_4s"],
                            ["RES4LYF res_4s ODE", "res_4s_ode"]),
    "res_2m": ([],
               ["Res Multistep", "k_res_multistep"]),
}


def _make_sampler(rk_type: str, ode: bool):
    if ode:
        def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable,
                                  rk_type=rk_type, eta=0.0)
    else:
        def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                    eta=1.0, s_noise=1.0):
            return sample_rk_beta(model, x, sigmas, None, extra_args, callback, disable,
                                  rk_type=rk_type, eta=eta, s_noise=s_noise)

    sampler.__name__ = f"sample_res4lyf_{rk_type.replace('-', '_')}{'_ode' if ode else ''}"
    return torch.no_grad()(sampler)


def _build():
    rows = []
    for rk_type, calls in METHODS:
        legacy_sde, legacy_ode = LEGACY.get(rk_type, ([], []))
        rows.append((f"RES4LYF {rk_type}", _make_sampler(rk_type, ode=False),
                     [rk_type, *legacy_sde], calls))
        rows.append((f"RES4LYF {rk_type} ODE", _make_sampler(rk_type, ode=True),
                     [f"{rk_type}_ode", *legacy_ode], calls))
    return rows


# (display name, function, aliases, model calls per step)
RES4LYF_SAMPLERS = _build()
