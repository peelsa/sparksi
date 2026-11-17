# SPDX-License-Identifier: Apache-2.0

"""Model inventory file labels + served-id display map. Read-only, cached by mtime."""

import os

from sparksi import compat, protect

try:
    import yaml
except ImportError:
    yaml = None

MODEL_INVENTORY = (
    os.environ.get("SPARKSI_MODEL_INVENTORY")
    or compat.legacy_env("SPARKSI_MODEL_INVENTORY")
    or compat.first_existing(compat.MODEL_INVENTORY_FALLBACKS)
)

# Fallback when the model inventory file is missing: generic seat labels only.
MODELS_FALLBACK = [
    {"name": "Brain", "role": "brain", "port": 8007},
    {"name": "Vision", "role": "vision", "port": 8004},
    {"name": "Counsel", "role": "counsel", "port": 8011},
]

# Kept so an inventory document with no ports still has the old label set.
# The live probe does not use this list; it curls whatever is listening.
PROBE_PORTS_EXTRA = [8000, 8001, 8009, 8013, 8014, 8015, 8016, 8888]  # plus any protect preset port

# Served id -> display name. Filled from the inventory file when it is read, and
# from SPARKSI_SERVED_DISPLAY=id=Name,id2=Name2 when set.
SERVED_DISPLAY = {
    k.strip(): v.strip()
    for k, _, v in (
        item.partition("=") for item in (os.environ.get("SPARKSI_SERVED_DISPLAY") or "").split(",") if "=" in item
    )
}

_cache = {"mtime": None, "models": None, "ports": None}


def models_from_text(text):
    """(models_list, ports_set) from a model inventory file body.

    Empty text falls back to the legacy label list. A real document with
    no `models:` entries stays empty so a stale fallback is not invented.
    """
    models = []
    ports = set(PROBE_PORTS_EXTRA) | protect.off_limits_ports()
    parsed = False
    if yaml is not None and text and str(text).strip():
        try:
            doc = yaml.safe_load(text) or {}
            parsed = True
            for m in doc.get("models") or []:
                port = m.get("port")
                if not port:
                    continue
                port = int(port)
                display = m.get("display") or m.get("served_name") or f":{port}"
                name = str(display).replace(" (NVFP4)", "").replace("Ultimate Uncensored", "Unc")
                if len(name) > 22:
                    name = name[:21] + "…"
                models.append({
                    "name": name,
                    "role": str(m.get("role") or "svc"),
                    "port": port,
                    "served_name": m.get("served_name"),
                    "unit": m.get("unit"),
                    "container": m.get("container"),
                    "display": str(display),
                })
                ports.add(port)
                sn = (m.get("served_name") or "").strip().lower()
                if sn and sn not in SERVED_DISPLAY:
                    SERVED_DISPLAY[sn] = name.split()[0][:22] if name else sn
        except Exception:
            models = []
            parsed = False
    if not models and not parsed:
        models = [dict(x) for x in MODELS_FALLBACK]
        for m in models:
            ports.add(m["port"])
    return models, ports


def load_known_models():
    """(models_list, ports_set) from the local model inventory file or fallback."""
    try:
        mtime = os.path.getmtime(MODEL_INVENTORY)
    except OSError:
        mtime = None
    if _cache["models"] is not None and _cache["mtime"] == mtime:
        return _cache["models"], _cache["ports"]
    text = ""
    if mtime is not None:
        try:
            with open(MODEL_INVENTORY, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = ""
    models, ports = models_from_text(text)
    _cache["mtime"] = mtime
    _cache["models"] = models
    _cache["ports"] = ports
    return models, ports
