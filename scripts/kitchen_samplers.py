# SPDX-License-Identifier: AGPL-3.0-or-later
# Kitchen Samplers. Third-party code and licenses: NOTICE.md

"""Kitchen Samplers: extra samplers and schedule types for Forge, reForge and Neo."""

import os
import sys
import traceback

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXTENSION_ROOT not in sys.path:
    sys.path.insert(0, EXTENSION_ROOT)

from modules import script_callbacks  # noqa: E402

from lib_kitchen import host, register, settings  # noqa: E402

LOG = register.LOG

# Extra Samplers registers the same unique samplers; running both doubles them.
CONFLICTS = {
    "sd-forge-extra-samplers": "Extra Samplers",
}


def _warn_replaced():
    try:
        from modules import extensions
        active = {e.name.lower(): e.name for e in extensions.active()}
    except Exception:
        return
    for folder, what in CONFLICTS.items():
        if folder in active:
            print(f"{LOG} {what} ({active[folder]}) is also enabled. "
                  f"Disable it in Extensions -> Installed, or both will register the same samplers.")


def on_before_ui():
    from lib_kitchen.catalogue import ALL
    from lib_kitchen.schedulers import SCHEDULERS

    _warn_replaced()
    steps = (
        ("schedule types", lambda: register.register_schedulers(SCHEDULERS)),
        ("samplers", lambda: register.register_samplers(ALL)),
        ("reForge scheduler fix", register.patch_reforge_alter_samplers),
        ("reForge schedule end fix", register.patch_reforge_schedulers),
        ("settings", settings.register_options),
        ("X/Y/Z axes", settings.add_xyz_axes),
    )
    for label, step in steps:
        try:
            step()
        except Exception:
            print(f"{LOG} failed to set up {label}:")
            traceback.print_exc()

    print(f"{LOG} {host.name()}: added {len(register.added_samplers)} samplers and "
          f"{len(register.added_schedulers)} schedule types; "
          f"{len(register.aliased_samplers)} were already here and kept as the WebUI's own.")


script_callbacks.on_before_ui(on_before_ui)
script_callbacks.on_infotext_pasted(settings.on_infotext_pasted)
