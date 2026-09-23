# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Settings → Kitchen Samplers, X/Y/Z axes, and Extra Samplers PNG-info keys."""

from __future__ import annotations

import inspect
import re

from lib_kitchen import params, register

SECTION = ("kitchen_samplers", "Kitchen Samplers")

# Keys match reForge's shared_options.py. Ranges skip values that crash (max_order 1, DEIS newton).
_S = "slider"
SPECS = {
    # samplers
    "heunpp2_s_churn": (0.0, "HeunPP2 - s_churn", _S, (-1.0, 2.0, 0.01), "HeunPP2 s_churn"),
    "heunpp2_s_tmin": (0.0, "HeunPP2 - s_tmin", _S, (-1.0, 2.0, 0.01), "HeunPP2 s_tmin"),
    "heunpp2_s_noise": (1.0, "HeunPP2 - s_noise", _S, (-1.0, 2.0, 0.1), "HeunPP2 s_noise"),
    "ipndm_max_order": (4, "IPNDM - max_order", _S, (2, 4, 1), "IPNDM max_order"),
    "ipndm_v_max_order": (4, "IPNDM-V - max_order", _S, (2, 4, 1), "IPNDM-V max_order"),
    "deis_max_order": (3, "DEIS - max_order", _S, (2, 4, 1), "DEIS max_order"),
    "deis_mode": ("tab", "DEIS - mode", "dropdown", ("tab",), "DEIS mode"),
    "dpmpp_2s_ancestral_cfg_pp_eta": (1.0, "DPM++ 2S Ancestral CFG++ - eta", _S, (-1.0001, 2.0, 0.0001), "DPM++ 2S Ancestral CFG++ eta"),
    "dpmpp_2s_ancestral_cfg_pp_s_noise": (1.0, "DPM++ 2S Ancestral CFG++ - s_noise", _S, (-1.0, 2.0, 0.1), "DPM++ 2S Ancestral CFG++ s_noise"),
    "dpmpp_sde_cfg_pp_eta": (1.0, "DPM++ SDE CFG++ - eta", _S, (-1.0001, 2.0, 0.0001), "DPM++ SDE CFG++ eta"),
    "dpmpp_sde_cfg_pp_s_noise": (1.0, "DPM++ SDE CFG++ - s_noise", _S, (-1.0, 2.0, 0.1), "DPM++ SDE CFG++ s_noise"),
    "dpmpp_sde_cfg_pp_r": (0.5, "DPM++ SDE CFG++ - r", _S, (-1.0, 2.0, 0.1), "DPM++ SDE CFG++ r"),
    "dpmpp_3m_sde_cfg_pp_eta": (1.0, "DPM++ 3M SDE CFG++ - eta", _S, (-1.0001, 2.0, 0.0001), "DPM++ 3M SDE CFG++ eta"),
    "dpmpp_3m_sde_cfg_pp_s_noise": (1.0, "DPM++ 3M SDE CFG++ - s_noise", _S, (-1.0, 2.0, 0.1), "DPM++ 3M SDE CFG++ s_noise"),
    "euler_dy_s_churn": (0.0, "Euler DY - s_churn", _S, (-1.0, 2.0, 0.01), "Euler DY s_churn"),
    "euler_dy_s_tmin": (0.0, "Euler DY - s_tmin", _S, (-1.0, 2.0, 0.01), "Euler DY s_tmin"),
    "euler_dy_s_noise": (1.0, "Euler DY - s_noise", _S, (-1.0, 2.0, 0.1), "Euler DY s_noise"),
    "euler_smea_dy_s_churn": (0.0, "Euler SMEA DY - s_churn", _S, (-1.0, 2.0, 0.01), "Euler SMEA DY s_churn"),
    "euler_smea_dy_s_tmin": (0.0, "Euler SMEA DY - s_tmin", _S, (-1.0, 2.0, 0.01), "Euler SMEA DY s_tmin"),
    "euler_smea_dy_s_noise": (1.0, "Euler SMEA DY - s_noise", _S, (-1.0, 2.0, 0.1), "Euler SMEA DY s_noise"),
    "euler_negative_s_churn": (0.0, "Euler Negative - s_churn", _S, (-1.0, 2.0, 0.01), "Euler Negative s_churn"),
    "euler_negative_s_tmin": (0.0, "Euler Negative - s_tmin", _S, (-1.0, 2.0, 0.01), "Euler Negative s_tmin"),
    "euler_negative_s_noise": (1.0, "Euler Negative - s_noise", _S, (-1.0, 2.0, 0.1), "Euler Negative s_noise"),
    "euler_dy_negative_s_churn": (0.0, "Euler DY Negative - s_churn", _S, (-1.0, 2.0, 0.01), "Euler DY Negative s_churn"),
    "euler_dy_negative_s_tmin": (0.0, "Euler DY Negative - s_tmin", _S, (-1.0, 2.0, 0.01), "Euler DY Negative s_tmin"),
    "euler_dy_negative_s_noise": (1.0, "Euler DY Negative - s_noise", _S, (-1.0, 2.0, 0.1), "Euler DY Negative s_noise"),
    # schedulers
    "sinusoidal_sf_factor": (3.5, "Sinusoidal SF scheduler - factor", _S, (0.1, 10.0, 0.1), "Sinusoidal SF scheduler factor"),
    "invcosinusoidal_sf_factor": (3.5, "Invcosinusoidal SF scheduler - factor", _S, (0.1, 10.0, 0.1), "Invcosinusoidal SF scheduler factor"),
    "react_cosinusoidal_dynsf_factor": (2.15, "React Cosinusoidal DynSF scheduler - factor", _S, (0.1, 10.0, 0.05), "React Cosinusoidal DynSF scheduler factor"),
    "cosine_sf_factor": (1.0, "Cosine scheduler - scale factor", _S, (0.1, 5.0, 0.1), "Cosine scheduler scale factor"),
    "cosexpblend_exp_decay": (0.9, "Cosine-exponential Blend scheduler - exponential decay", _S, (0.1, 0.99, 0.01), "Cosine-exponential Blend scheduler exponential decay"),
    "phi_power": (2.0, "Phi scheduler - power", _S, (1.0, 5.0, 0.1), "Phi scheduler power"),
    "laplace_mu": (0.0, "Laplace scheduler - mu", _S, (-1.0, 1.0, 0.1), "Laplace scheduler mu"),
    "laplace_beta": (0.5, "Laplace scheduler - beta", _S, (0.1, 2.0, 0.1), "Laplace scheduler beta"),
    "karras_dynamic_rho": (7.0, "Karras Dynamic scheduler - base rho", _S, (1.0, 20.0, 0.1), "Karras Dynamic scheduler base rho"),
    "ays_custom_sigmas": ("[14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]",
                          "Align Your Steps Custom - sigma values", "text", (), "AYS Custom sigmas"),
}

_OPT_RE = re.compile(r"_host\.opt\(\s*['\"]([a-zA-Z0-9_]+)['\"]")


def options_read_by(func) -> list[str]:
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return []
    return list(dict.fromkeys(_OPT_RE.findall(src)))


def wanted_options():
    """(Param-style entries, SPECS keys) for what was actually added this run."""
    from lib_kitchen.catalogue import ALL
    from lib_kitchen.schedulers import SCHEDULERS

    ours, spec_keys = [], []
    added = set(register.added_samplers)
    for e in ALL:
        if e.label not in added:
            continue
        ours.extend(e.settings)
        spec_keys.extend(k for k in options_read_by(e.func) if k in SPECS)
    added_sched = set(register.added_schedulers)
    for name, label, fn, _inner, _aliases, keys in SCHEDULERS:
        if label in added_sched:
            spec_keys.extend(k for k in keys if k in SPECS)
    return list(dict.fromkeys(ours)), list(dict.fromkeys(spec_keys))


def _component(kind, args):
    import gradio as gr
    if kind == "slider":
        lo, hi, step = args
        return gr.Slider, {"minimum": lo, "maximum": hi, "step": step}
    if kind == "dropdown":
        return gr.Dropdown, {"choices": list(args)}
    return gr.Textbox, {}


_ever_added: set[str] = set()


def register_options():
    """Register settings for samplers/schedules added this run. Called from on_before_ui so X/Y/Z and API-only see them."""
    from modules import shared

    defined = getattr(shared.opts, "data_labels", {})
    register.registered_options.clear()
    ours, spec_keys = wanted_options()

    def add(key, info):
        if key in defined and key not in _ever_added:
            return                      # the WebUI (or another extension) owns it
        _ever_added.add(key)
        info.section = SECTION
        if getattr(info, "category_id", None) is None:
            info.category_id = "sd"
        shared.opts.add_option(key, info)
        register.registered_options.add(key)

    for p in ours:
        comp, args = _component("slider", (p.minimum, p.maximum, p.step))
        info = shared.OptionInfo(p.default, p.label, comp, args, infotext=p.infotext)
        if p.info:
            info = info.info(p.info)
        add(p.key, info)

    for key in spec_keys:
        default, label, kind, args, infotext = SPECS[key]
        comp, cargs = _component(kind, args)
        add(key, shared.OptionInfo(default, label, comp, cargs, infotext=infotext))


def on_infotext_pasted(infotext, params_dict):
    """PNG info from Extra Samplers named its values by their Settings key."""
    from lib_kitchen.params import ADAPTIVE_PROGRESSIVE, LANGEVIN
    for p in (*ADAPTIVE_PROGRESSIVE, *LANGEVIN):
        for old in p.legacy_infotext:
            if old in params_dict and p.infotext not in params_dict:
                params_dict[p.infotext] = params_dict[old]


# ------------------------------------------------------------------ X/Y/Z

AXIS_PREFIX = "[Kitchen] "


def add_xyz_axes():
    """One X/Y/Z axis per registered setting, applied through override_settings."""
    from modules import scripts, shared

    xyz = None
    for data in scripts.scripts_data:
        if data.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid") and hasattr(data, "module"):
            xyz = data.module
            break
    if xyz is None:
        return 0

    xyz.axis_options[:] = [a for a in xyz.axis_options if not str(a.label).startswith(AXIS_PREFIX)]
    labels = getattr(shared.opts, "data_labels", {})
    count = 0
    for key in sorted(register.registered_options):
        info = labels.get(key)
        if info is None:
            continue
        default = info.default
        kind = float if isinstance(default, float) else int if isinstance(default, int) and not isinstance(default, bool) else str
        xyz.axis_options.append(xyz.AxisOption(AXIS_PREFIX + info.label, kind, xyz.apply_override(key)))
        count += 1
    return count
