# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Register catalogue entries on the WebUI. Strips a previous pass first so Reload UI does not duplicate them."""

from __future__ import annotations

import re

from lib_kitchen import host

LOG = "[Kitchen Samplers]"
MARK = "_kitchen_samplers"          # set in SamplerData.options of everything added here

added_samplers: list[str] = []
aliased_samplers: dict[str, str] = {}   # our label -> the WebUI's entry it resolved to
added_schedulers: list[str] = []
registered_options: set[str] = set()


def _norm(name: str) -> str:
    """Case and punctuation do not make a different sampler: 'SA-Solver' == 'SA Solver'."""
    return re.sub(r"[^a-z0-9+]", "", str(name).lower())


# ------------------------------------------------------------------ samplers

def _ours(data) -> bool:
    return isinstance(getattr(data, "options", None), dict) and data.options.get(MARK, False)


def _sampler_data_class(options: dict):
    """SamplerData, or a subclass whose total_steps counts every model call (multi-stage / start-up)."""
    from modules import sd_samplers_common

    if int(options.get("stages", 1)) < 3 and not options.get("extra_calls"):
        return sd_samplers_common.SamplerData

    class StagedSamplerData(sd_samplers_common.SamplerData):
        def total_steps(self, steps):
            per_step = int(self.options.get("stages", 2 if self.options.get("second_order") else 1))
            return steps * per_step + int(self.options.get("extra_calls", 0))

    return StagedSamplerData


def register_samplers(entries) -> None:
    from modules import sd_samplers, sd_samplers_kdiffusion
    from modules.sd_samplers_kdiffusion import KDiffusionSampler

    added_samplers.clear()
    aliased_samplers.clear()

    # 1. strip what a previous pass added
    for seq_name in ("all_samplers", "samplers_data_k_diffusion"):
        seq = getattr(sd_samplers, seq_name, None) if seq_name == "all_samplers" else getattr(
            sd_samplers_kdiffusion, seq_name, None)
        if isinstance(seq, list):
            seq[:] = [s for s in seq if not _ours(s)]

    # 2. what the WebUI (and any other extension) already lists, by normalised name
    listed = {}
    for s in sd_samplers.all_samplers:
        for n in [s.name, *(s.aliases or [])]:
            listed.setdefault(_norm(n), s.name)

    extra_params = getattr(sd_samplers_kdiffusion, "sampler_extra_params", None)

    for e in entries:
        match = next((listed[_norm(n)] for n in e.names if _norm(n) in listed), None)
        if match is not None:
            aliased_samplers[e.label] = match
            continue

        options = dict(e.options)
        options[MARK] = True
        cls = _sampler_data_class(options)
        taken = {_norm(n) for n in listed}
        aliases = [a for a in dict.fromkeys(e.also) if _norm(a) not in taken and _norm(a) != _norm(e.label)]

        data = cls(e.label,
                   (lambda model, _f=e.func, _o=options: KDiffusionSampler(_f, model, _o)),
                   aliases, options)
        sd_samplers.all_samplers.append(data)
        if isinstance(extra_params, dict) and e.extra_params:
            extra_params[e.func] = list(e.extra_params)

        for n in [e.label, *aliases]:
            listed.setdefault(_norm(n), e.label)
        added_samplers.append(e.label)

    _refresh_sampler_tables()


def _refresh_sampler_tables():
    """Rebuild all_samplers_map / samplers_map and clear get_sampler_and_scheduler's cache."""
    from modules import sd_samplers

    table = {s.name: s for s in sd_samplers.all_samplers}
    for s in sd_samplers.all_samplers:
        if _ours(s):
            for alias in s.aliases:
                table.setdefault(alias, s)
    by_name = {s.name: s for s in sd_samplers.all_samplers}
    from lib_kitchen.catalogue import ALL
    for e in ALL:
        target = aliased_samplers.get(e.label)
        if target in by_name:
            for n in e.names:
                table.setdefault(n, by_name[target])

    current = getattr(sd_samplers, "all_samplers_map", None)
    if isinstance(current, dict):
        current.clear()
        current.update(table)
    else:
        sd_samplers.all_samplers_map = table

    sd_samplers.set_samplers()
    # set_samplers() filled samplers_map from names and aliases; add the
    # cross-WebUI names too, so a lower-cased lookup finds them.
    samplers_map = getattr(sd_samplers, "samplers_map", None)
    if isinstance(samplers_map, dict):
        for n, s in table.items():
            samplers_map.setdefault(n.lower(), s.name)
    cache_clear = getattr(getattr(sd_samplers, "get_sampler_and_scheduler", None), "cache_clear", None)
    if cache_clear:
        cache_clear()


# ------------------------------------------------------------------ schedulers

def register_schedulers(rows) -> None:
    from modules import shared, sd_schedulers

    registry = sd_schedulers.schedulers
    every = getattr(sd_schedulers, "all_schedulers", None)
    mapping = sd_schedulers.schedulers_map

    # strip a previous pass
    for seq in (registry, every):
        if isinstance(seq, list):
            seq[:] = [s for s in seq if not getattr(s, MARK, False)]
    for key in [k for k, v in mapping.items() if getattr(v, MARK, False)]:
        del mapping[key]

    added_schedulers.clear()
    hidden = set(getattr(shared.opts, "hide_schedulers", None) or [])
    known = set()
    for s in (every if isinstance(every, list) else registry):
        known.update(_norm(n) for n in [s.name, s.label, *(getattr(s, "aliases", None) or [])])

    for name, label, function, need_inner, aliases, _opts in rows:
        if any(_norm(n) in known for n in [name, label, *aliases]):
            continue                       # the WebUI has it: never shadow its version
        entry = sd_schedulers.Scheduler(name, label, function, need_inner_model=need_inner,
                                        aliases=list(aliases) or None)
        setattr(entry, MARK, True)
        if isinstance(every, list):
            every.append(entry)
        if label not in hidden:
            registry.append(entry)
            for key in [name, label, *aliases]:
                mapping.setdefault(key, entry)
        known.update(_norm(n) for n in [name, label, *aliases])
        added_schedulers.append(label)


# ------------------------------------------------------------------ reForge fixes

# reForge copies that omit the final 0 (one step short).
_SHORT_SCHEDULES = ("Sinusoidal SF", "Invcosinusoidal SF", "React Cosinusoidal DynSF")


def patch_reforge_schedulers() -> list[str]:
    """Append a final 0 to reForge's own short copies of the schedules above."""
    if host.name() != "reforge":
        return []
    import torch
    from modules import sd_schedulers

    fixed = []
    for s in [*sd_schedulers.schedulers, *sd_schedulers.schedulers_map.values()]:
        if s.label not in _SHORT_SCHEDULES or getattr(s, MARK, False) or getattr(s.function, MARK, False):
            continue
        original = s.function

        def ends_at_zero(*args, _original=original, **kwargs):
            sigmas = _original(*args, **kwargs)
            if sigmas.numel() and float(sigmas[-1]) != 0.0:
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
            return sigmas

        ends_at_zero.__name__ = getattr(original, "__name__", "schedule")
        ends_at_zero.__wrapped__ = original
        setattr(ends_at_zero, MARK, True)
        s.function = ends_at_zero
        fixed.append(s.label)
    try:
        from modules import sd_samplers_kdiffusion as sk
        table = getattr(sk, "k_diffusion_scheduler", None)
        if isinstance(table, dict):
            for s in sd_schedulers.schedulers:
                if s.name in table:
                    table[s.name] = s.function
    except Exception:
        pass
    return sorted(set(fixed))


def patch_reforge_alter_samplers() -> bool:
    """Let reForge alter-samplers use schedule types this extension added, instead of falling back to Normal."""
    try:
        from modules import sd_samplers, sd_samplers_kdiffusion
        from modules_forge import forge_alter_samplers as fas
    except Exception:
        return False
    alter = getattr(fas, "AlterSampler", None)
    if alter is None:
        return False

    def ours(label) -> bool:
        return label in added_schedulers

    if not getattr(alter.get_sigmas, MARK, False):
        original_get_sigmas = alter.get_sigmas

        def get_sigmas(self, p, steps):
            label = (p.hr_scheduler if getattr(p, "is_hr_pass", False) else p.scheduler) or self.scheduler_name
            if ours(label):
                sigmas = sd_samplers_kdiffusion.KDiffusionSampler.get_sigmas(self, p, steps)
                return sigmas.to(self.unet.load_device)
            return original_get_sigmas(self, p, steps)

        setattr(get_sigmas, MARK, True)
        alter.get_sigmas = get_sigmas

    current = sd_samplers.get_sampler_and_scheduler
    if not getattr(current, MARK, False):
        original = current

        def get_sampler_and_scheduler(sampler_name, scheduler_name, *, convert_automatic=True):
            sampler, scheduler = original(sampler_name, scheduler_name, convert_automatic=convert_automatic)
            if scheduler_name and scheduler != scheduler_name:
                from modules import sd_schedulers
                found = sd_schedulers.schedulers_map.get(scheduler_name)
                is_alter = any(s.name.lower() == str(sampler).lower() for s in fas.samplers_data_alter)
                if is_alter and found is not None and ours(found.label):
                    return sampler, found.label
            return sampler, scheduler

        # Keep cache_clear on the wrapper so table refresh can still clear it.
        get_sampler_and_scheduler.cache_clear = getattr(original, "cache_clear", lambda: None)
        get_sampler_and_scheduler.__wrapped__ = original
        setattr(get_sampler_and_scheduler, MARK, True)
        sd_samplers.get_sampler_and_scheduler = get_sampler_and_scheduler
    return True
