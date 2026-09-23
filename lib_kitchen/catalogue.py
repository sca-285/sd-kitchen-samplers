# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Sampler catalogue. Each algorithm is listed once; other names become aliases if the WebUI already has it."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

from lib_kitchen.canon import from_neo as N
from lib_kitchen.canon import from_reforge as R
from lib_kitchen.canon import smea as SMEA
from lib_kitchen.extra import (adaptive_progressive, euler_max, euler_multipass, euler_smea,
                               euler_smea_dy_negative, heun_ancestral, langevin_euler,
                               sa_solver_stable, ssprk3)
from lib_kitchen.res4lyf.table import RES4LYF_SAMPLERS, STARTUP_CALLS


@dataclass
class Entry:
    label: str
    func: Callable
    also: list = field(default_factory=list)      # other names for the same algorithm
    options: dict = field(default_factory=dict)   # SamplerData options (scheduler, flags)
    cost: int = 1
    source: str = ""                              # where the code comes from (for the README)
    settings: tuple = ()                          # lib_kitchen.params entries it reads

    @property
    def names(self):
        return [self.label, *self.also]


def _extra_params(func):
    """s_churn / s_tmin / s_tmax / s_noise present on the function signature."""
    wanted = ("s_churn", "s_tmin", "s_tmax", "s_noise")
    params = inspect.signature(func).parameters
    return [p for p in wanted if p in params]


# ---------------------------------------------------------------- WebUI samplers

BROWNIAN = {"brownian_noise": True}

WEBUI = [
    Entry("DEIS", R.sample_deis, ["deis", "k_deis"], source="reForge / Forge"),
    Entry("IPNDM", R.sample_ipndm, ["iPNDM", "ipndm", "k_ipndm"], source="reForge / Forge"),
    Entry("IPNDM_V", R.sample_ipndm_v, ["iPNDM_v", "IPNDM V", "ipndm_v", "k_ipndm_v"], source="reForge / Forge"),
    Entry("HeunPP2", R.sample_heunpp2, ["Heun++ 2", "heunpp2", "k_heunpp2"], {"stages": 3}, cost=3,
          source="reForge / Forge"),
    Entry("DDPM", R.sample_ddpm, ["ddpm", "k_ddpm"], dict(BROWNIAN), source="reForge / Forge"),

    Entry("Res Multistep", N.sample_res_multistep, ["RES Multistep", "res_multistep", "k_res_multistep"],
          source="Neo / reForge"),
    Entry("Res Multistep CFG++", R.sample_res_multistep_cfg_pp,
          ["RES Multistep CFG++", "res_multistep_cfg_pp", "k_res_multistep_cfg_pp"], source="reForge"),
    Entry("Res Multistep Ancestral", R.sample_res_multistep_ancestral,
          ["RES Multistep Ancestral", "res_multistep_ancestral", "k_res_multistep_a"],
          {"uses_ensd": True}, source="reForge"),
    Entry("Res Multistep Ancestral CFG++", R.sample_res_multistep_ancestral_cfg_pp,
          ["RES Multistep Ancestral CFG++", "res_multistep_ancestral_cfg_pp", "k_res_multistep_a_cfg_pp"],
          {"uses_ensd": True}, source="reForge"),

    Entry("Gradient Estimation", R.sample_gradient_estimation,
          ["gradient_estimation", "k_gradient_estimation"], source="reForge"),
    Entry("Gradient Estimation CFG++", R.sample_gradient_estimation_cfg_pp,
          ["gradient_estimation_CFG++", "gradient_estimation_cfg_pp", "k_gradient_estimation_cfg_pp"],
          source="reForge"),

    Entry("ER SDE", N.sample_er_sde, ["er_sde", "k_er_sde"], {"scheduler": "exponential", **BROWNIAN},
          source="Neo / reForge"),
    Entry("SEEDS 2 DP", R.sample_seeds_2, ["SEEDS 2", "SEEDS-2", "SEEDS_2", "seeds_2", "k_seeds_2"],
          {"second_order": True, "uses_ensd": True, **BROWNIAN}, cost=2, source="reForge"),
    Entry("SEEDS 3 DP", R.sample_seeds_3, ["SEEDS 3", "SEEDS-3", "SEEDS_3", "seeds_3", "k_seeds_3"],
          {"stages": 3, "uses_ensd": True, **BROWNIAN}, cost=3, source="reForge"),
    Entry("SA-Solver", R.sample_sa_solver, ["SA Solver", "sa_solver", "k_sa_solver"],
          {"uses_ensd": True, **BROWNIAN}, source="reForge"),
    Entry("SA-Solver-Pece", R.sample_sa_solver_pece,
          ["SA-Solver PECE", "SA Solver PECE", "sa_solver_pece", "k_sa_solver_pece"],
          {"second_order": True, "uses_ensd": True, **BROWNIAN}, cost=2, source="reForge"),

    Entry("Euler CFG++", N.sample_euler_cfg_pp, ["euler_cfg_pp", "k_euler_cfg_pp"], source="Neo / reForge"),
    Entry("Euler a CFG++", N.sample_euler_ancestral_cfg_pp,
          ["Euler Ancestral CFG++", "euler_ancestral_cfg_pp", "k_euler_a_cfg_pp"], {"uses_ensd": True},
          source="Neo / reForge"),
    Entry("DPM++ 2M CFG++", N.sample_dpmpp_2m_cfg_pp, ["dpmpp_2m_cfg_pp", "k_dpmpp_2m_cfg_pp"],
          source="Neo / reForge"),
    Entry("DPM++ 2S a CFG++", R.sample_dpmpp_2s_ancestral_cfg_pp,
          ["DPM++ 2S Ancestral CFG++", "dpmpp_2s_ancestral_cfg_pp", "k_dpmpp_2s_a_cfg_pp"],
          {"second_order": True, "uses_ensd": True}, cost=2, source="reForge"),
    Entry("DPM++ SDE CFG++", R.sample_dpmpp_sde_cfg_pp, ["dpmpp_sde_cfg_pp", "k_dpmpp_sde_cfg_pp"],
          {"second_order": True, **BROWNIAN}, cost=2, source="reForge"),
    Entry("DPM++ 3M SDE CFG++", R.sample_dpmpp_3m_sde_cfg_pp, ["dpmpp_3m_sde_cfg_pp", "k_dpmpp_3m_sde_cfg_pp"],
          {"scheduler": "exponential", "discard_next_to_last_sigma": True, **BROWNIAN}, source="reForge"),

    Entry("Euler Dy", SMEA.sample_euler_dy, ["euler_dy", "k_euler_dy"], source="reForge (Koishi-Star)"),
    Entry("Euler SMEA Dy", SMEA.sample_euler_smea_dy, ["euler_smea_dy", "k_euler_smea_dy"],
          source="reForge (Koishi-Star)"),
    Entry("Euler Negative", SMEA.sample_euler_negative, ["euler_negative", "k_euler_negative"],
          source="reForge (Koishi-Star)"),
    Entry("Euler Negative Dy", SMEA.sample_euler_dy_negative,
          ["Euler Dy Negative", "euler_dy_negative", "k_euler_dy_negative"], source="reForge (Koishi-Star)"),
]

# ---------------------------------------------------------------- Extra Samplers (MisterChief95). WebUI copies of the same algorithms are not used.

from lib_kitchen import params  # noqa: E402

EXTRA = [
    Entry("Adaptive Progressive", adaptive_progressive.sample_adaptive_progressive, ["adaptive_progressive"],
          {"scheduler": "sgm_uniform", "uses_ensd": True}, source="Extra Samplers",
          settings=params.ADAPTIVE_PROGRESSIVE),
    Entry("Euler Max", euler_max.sample_euler_max, ["euler_max"], source="Extra Samplers (licyk)"),
    Entry("Euler SMEA", euler_smea.sample_euler_smea, ["euler_smea"], source="Extra Samplers (licyk)"),
    Entry("Euler SMEA Dy Negative", euler_smea_dy_negative.sample_euler_smea_dy_negative,
          ["euler_smea_dy_negative"], source="Extra Samplers (Koishi-Star)"),
    Entry("Euler Multipass", euler_multipass.sample_euler_multipass, ["euler_multipass"],
          source="Extra Samplers (aria1th, catboxanon)"),
    Entry("Euler Multipass CFG++", euler_multipass.sample_euler_multipass_cfg_pp, ["euler_multipass_cfg++"],
          source="Extra Samplers (aria1th, LaVie024)"),
    Entry("Euler a Multipass", euler_multipass.sample_euler_ancestral_multipass, ["euler_a_multipass"],
          {"uses_ensd": True}, source="Extra Samplers (aria1th)"),
    Entry("Euler a Multipass CFG++", euler_multipass.sample_euler_ancestral_multipass_cfg_pp,
          ["euler_a_multipass_cfg++"], {"uses_ensd": True}, source="Extra Samplers (aria1th, LaVie024)"),
    # brownian_noise so the WebUI supplies seeded noise (upstream used extra_args["seed"], which no WebUI sets).
    Entry("Heun Ancestral", heun_ancestral.sample_heun_ancestral, ["heun_ancestral"],
          {"second_order": True, "uses_ensd": True, **BROWNIAN}, cost=2, source="Extra Samplers"),
    Entry("Langevin Euler", langevin_euler.sample_langevin_euler, ["langevin_euler"],
          {"scheduler": "sgm_uniform"}, source="Extra Samplers", settings=params.LANGEVIN),
    Entry("SSPRK3", ssprk3.sample_ssprk3, ["ssprk3"], {"stages": 3}, cost=3, source="Extra Samplers"),
    Entry("SA-Solver Stable", sa_solver_stable.sample_sa_solver_stable, ["sa_solver_stable"],
          {"scheduler": "karras"}, source="Extra Samplers (kabachuha, eddyhhlure1Eddy)"),
]

# ---------------------------------------------------------------- RES4LYF
# res_2m ODE == Res Multistep; res_2s_stable ODE == EXP Heun 2 x0. Noisy res_2m is not Res Multistep Ancestral.
SAME_AS_WEBUI = {
    "RES4LYF res_2m ODE": ["Res Multistep", "RES Multistep"],
    "RES4LYF res_2s_stable ODE": ["EXP Heun 2 x0"],
}

RES4LYF = []
for label, func, aliases, calls in RES4LYF_SAMPLERS:
    extra = list(aliases) + SAME_AS_WEBUI.get(label, [])
    options = {"scheduler": "sgm_uniform"}
    if calls >= 3:
        options["stages"] = calls
    elif calls == 2:
        options["second_order"] = True
    startup = STARTUP_CALLS.get(label.split()[1], 0)
    if startup:
        options["extra_calls"] = startup
    RES4LYF.append(Entry(label, func, extra, options, cost=calls, source="RES4LYF (ClownsharkBatwing), rewritten"))


ALL = [*WEBUI, *EXTRA, *RES4LYF]

for _e in ALL:
    _e.extra_params = _extra_params(_e.func)
