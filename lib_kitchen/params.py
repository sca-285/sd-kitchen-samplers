# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Sampler settings read at runtime and written into PNG infotext."""

from __future__ import annotations

from dataclasses import dataclass, field

from lib_kitchen import host


@dataclass(frozen=True)
class Param:
    key: str                # the Settings key
    default: object
    label: str              # shown in Settings
    infotext: str           # written to PNG info
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple = ()
    info: str = ""
    legacy_infotext: tuple = field(default_factory=tuple)   # older names that should still paste


# Extra Samplers keys kept so config.json / old PNG info still apply.
AP_EULER_A_END = Param("exs_ap_euler_a_end", 0.35, "Adaptive Progressive - Euler a phase end",
                       "Adaptive Progressive Euler a end", 0.0, 1.0, 0.05,
                       info="Fraction of the run spent in the Euler a phase",
                       legacy_infotext=("exs_ap_euler_a_end",))
AP_DPM_2M_END = Param("exs_ap_dpm_2m_end", 0.75, "Adaptive Progressive - DPM++ 2M phase end",
                      "Adaptive Progressive DPM++ 2M end", 0.0, 1.0, 0.05,
                      info="Where the DPM++ 2M phase hands over to the detail phase",
                      legacy_infotext=("exs_ap_dpm_2m_end",))
AP_ANCESTRAL_ETA = Param("exs_ap_ancestral_eta", 0.4, "Adaptive Progressive - ancestral eta",
                         "Adaptive Progressive eta", 0.0, 1.0, 0.05,
                         legacy_infotext=("exs_ap_ancestral_eta",))
AP_DETAIL_STRENGTH = Param("exs_ap_detail_strength", 1.5, "Adaptive Progressive - detail strength",
                           "Adaptive Progressive detail strength", 0.0, 3.0, 0.05,
                           legacy_infotext=("exs_ap_detail_strength",))
LANGEVIN_STRENGTH = Param("exs_langevin_strength", 0.1, "Langevin Euler - strength",
                          "Langevin Euler strength", 0.0, 0.5, 0.01,
                          info="How much Langevin noise is mixed in each step",
                          legacy_infotext=("exs_langevin_strength",))

ADAPTIVE_PROGRESSIVE = (AP_EULER_A_END, AP_DPM_2M_END, AP_ANCESTRAL_ETA, AP_DETAIL_STRENGTH)
LANGEVIN = (LANGEVIN_STRENGTH,)


def read(model, param: Param):
    """The current value of `param`, recorded in the image's PNG info."""
    value = host.opt(param.key, param.default)
    p = getattr(model, "p", None)
    extra = getattr(p, "extra_generation_params", None)
    if isinstance(extra, dict):
        extra[param.infotext] = value
    return value
