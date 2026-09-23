# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Host differences between Forge, reForge and Neo (model sampling, noise, CFG++ hooks)."""

from __future__ import annotations

import contextlib
import functools
import inspect
import sys

import torch


# --------------------------------------------------------------------- which one

def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def name() -> str:
    """'reforge', 'neo' or 'forge' (for messages and tests; logic uses features)."""
    if _importable("ldm_patched.modules.model_sampling"):
        return "reforge"
    if _importable("modules_forge.packages.k_diffusion.sampling"):
        return "neo"
    return "forge"


# --------------------------------------------------------------------- the model

def model_sampling(model):
    """The object's sigma table / prediction type. `model` is the host CFG denoiser."""
    inner = getattr(model, "inner_model", model)
    predictor = getattr(inner, "predictor", None)
    if predictor is not None:                       # Forge, Neo
        return predictor
    patcher = getattr(inner, "model_patcher", None)
    if patcher is not None:                         # reForge
        return patcher.get_model_object("model_sampling")
    deeper = getattr(inner, "inner_model", None)
    for candidate in (inner, deeper, getattr(deeper, "forge_objects", None)):
        ms = getattr(candidate, "model_sampling", None)
        if ms is not None:
            return ms
    raise AttributeError("could not find the model's sampling object on this WebUI")


def is_const(ms) -> bool:
    """True for flow-matching / rectified-flow models (Flux, SD3, Wan...)."""
    if getattr(ms, "prediction_type", None) == "const":
        return True
    return any(cls.__name__ == "CONST" for cls in type(ms).__mro__)


def is_flow_model(model) -> bool:
    try:
        return is_const(model_sampling(model))
    except Exception:
        return False


# --------------------------------------------------------------------- noise

def _hijacked_torch():
    """The seeded TorchHijack installed on the host k-diffusion module, else plain torch."""
    for mod_name in ("ldm_patched.k_diffusion.sampling", "k_diffusion.sampling",
                     "modules_forge.packages.k_diffusion.sampling"):
        mod = sys.modules.get(mod_name)
        t = getattr(mod, "torch", None) if mod is not None else None
        if t is not None and t is not torch:
            return t
    return torch


def randn_like(x: torch.Tensor) -> torch.Tensor:
    return _hijacked_torch().randn_like(x)


# --------------------------------------------------------------------- options

def opt(key: str, default):
    """A Settings value, or `default` if this WebUI does not define it (yet)."""
    try:
        from modules import shared
        value = getattr(shared.opts, key)
    except Exception:
        return default
    return default if value is None else value


# --------------------------------------------------------------------- post-CFG

@functools.lru_cache(maxsize=None)
def cfg_denoiser_merges_model_options() -> bool:
    """True if the host CFG denoiser honours extra_args["model_options"]."""
    try:
        from modules.sd_samplers_cfg_denoiser import CFGDenoiser
        return "model_options" in inspect.getsource(CFGDenoiser.forward)
    except Exception:
        return True


_scope: list[list] = []


def scoped(fn):
    """Undo any UNet patch made by `with_post_cfg` when `fn` returns or raises."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _scope.append([])
        try:
            return fn(*args, **kwargs)
        finally:
            for undo in reversed(_scope.pop()):
                try:
                    undo()
                except Exception:
                    pass
    return wrapper


def _unet_patcher(model):
    inner = getattr(model, "inner_model", None)
    sd_model = getattr(inner, "inner_model", None)
    return getattr(getattr(sd_model, "forge_objects", None), "unet", None)


def with_post_cfg(model, extra_args: dict, post_cfg_function) -> dict:
    """Attach a post-CFG hook and disable the CFG=1 shortcut. CFG++ needs the uncond prediction."""
    extra_args = dict(extra_args or {})
    if cfg_denoiser_merges_model_options():
        options = dict(extra_args.get("model_options") or {})
        options["sampler_post_cfg_function"] = list(options.get("sampler_post_cfg_function", [])) + [post_cfg_function]
        options["disable_cfg1_optimization"] = True
        extra_args["model_options"] = options
        return extra_args

    unet = _unet_patcher(model)
    if unet is None:
        raise RuntimeError("CFG++: could not reach the UNet patcher on this WebUI")
    previous = unet.model_options
    options = dict(previous)
    options["sampler_post_cfg_function"] = list(previous.get("sampler_post_cfg_function", [])) + [post_cfg_function]
    options["disable_cfg1_optimization"] = True
    unet.model_options = options

    def undo():
        unet.model_options = previous

    if _scope:
        _scope[-1].append(undo)
    else:  # not inside a @scoped sampler: nothing will undo it, so say so
        raise RuntimeError("with_post_cfg on this WebUI must be called from a @scoped sampler")
    return extra_args


@contextlib.contextmanager
def post_cfg(model, extra_args: dict, post_cfg_function):
    """Context-manager form of with_post_cfg, usable outside @scoped."""
    _scope.append([])
    try:
        yield with_post_cfg(model, extra_args, post_cfg_function)
    finally:
        for undo in reversed(_scope.pop()):
            undo()
