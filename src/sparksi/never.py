# SPDX-License-Identifier: Apache-2.0

"""Hard refusals: mutators, forbidden binds, scripts we must not invoke."""

from __future__ import annotations

import re

MUTATOR_VERBS = ("load", "evict", "install")

REFUSE_MSG = (
    "refuses {verb!r}: load/evict/install are not implemented. "
    "Use apply (swap is an apply alias). No mutation performed."
)

# Site tmux sessions to leave alone are listed in the catalog (never_touch.tmux_sessions).
FORBIDDEN_TMUX = ("1",)

FORBIDDEN_INVOKE = (
    "video-mode.sh",
    "vllm-coresident-start.sh",
    "vllm-coresident-stop.sh",
    "tmux",
)

# Policy stays in code; concrete LAN/WiFi addresses come from config.
BIND_DENY_DEFAULT = ("0.0.0.0", "::")

# Front door 8000 is later. Trio / proxy ports are not ours to steal; a protect preset adds its own.
PORT_DENY_DEFAULT = (8000, 8001, 8004, 8007, 8011)

NVIDIA_SMI_ALLOW_PREFIX = (
    "--query-",
    "--format",
    "--list-gpus",
)
NVIDIA_SMI_ALLOW_EXACT = {"-L", "-h", "--help", "--format=csv,noheader,nounits", "-q", "--query"}
NVIDIA_SMI_DISPLAY_SECTIONS = frozenset({
    "CLOCK",
    "CLOCKS",
    "SUPPORTED_CLOCKS",
    "UTILIZATION",
    "TEMPERATURE",
    "PERFORMANCE",
})
NVIDIA_SMI_DENY_SUBSTR = (
    "reset",
    "persistenced",
    "persistence-mode",
    "power-limit",
    "gpu-reset",
    "applications-clocks",
    "lock-gpu-clocks",
    "lock-memory-clocks",
)

# Only a non-negative integer may appear as a GPU index or a max MHz in a cap command.
CLOCK_LOCK_INT = re.compile(r"(?:0|[1-9][0-9]*)")


class Refused(RuntimeError):
    pass


def refuse_mutator(verb: str) -> str:
    return REFUSE_MSG.format(verb=verb)


def launcher_refusal(argv: list[str] | tuple[str, ...]) -> str | None:
    """Block trio and unknown video-mode calls. The `video` load and `stop-video` are allowed."""
    parts = [str(p) for p in argv]
    if not parts:
        return None
    joined = " ".join(parts)
    base = parts[0].rstrip("/").rsplit("/", 1)[-1]
    if base == "tmux" or "vllm-coresident-" in joined:
        return "refusing forbidden command"
    mentions = base == "video-mode.sh" or any(p.rstrip("/").rsplit("/", 1)[-1] == "video-mode.sh" for p in parts)
    if not mentions:
        return None
    if len(parts) >= 2 and parts[-1] in ("stop-video", "video"):
        return None
    if parts[-1] == "trio":
        return "refusing video-mode.sh trio"
    return "refusing forbidden command"


def assert_bind_allowed(hosts: list[str], deny: list[str] | None = None) -> None:
    if deny is None:
        from sparksi.config import CFG

        deny = list(BIND_DENY_DEFAULT) + CFG.node_bind_deny()
    deny_set = {h.lower() for h in (deny or BIND_DENY_DEFAULT)}
    deny_set.update({"0.0.0.0", "::", "*"})
    for h in hosts:
        if h.lower() in deny_set:
            raise Refused(f"refusing bind {h!r}: localhost/Tailscale only")


def assert_serve_port(port: int, extra_deny: list[int] | None = None) -> None:
    from sparksi import protect

    deny = set(PORT_DENY_DEFAULT) | set(protect.off_limits_ports())
    if extra_deny:
        deny.update(int(p) for p in extra_deny)
    p = int(port)
    if p in deny:
        raise Refused(
            f"refusing serve --port {p}: reserved "
            f"({', '.join(str(x) for x in sorted(deny))}). Default is 7500. "
            "Front door 8000 is later."
        )
    if p < 1 or p > 65535:
        raise Refused(f"refusing serve --port {p}: invalid")


def nvidia_smi_is_readonly(rest: list[str]) -> bool:
    """Query/list flags only. Refuse reset / persistenced / any mutate argv."""
    if not rest:
        return False
    i = 0
    while i < len(rest):
        a = rest[i]
        al = a.lower()
        if a in ("-r", "--gpu-reset", "--reset", "-lgc", "-rgc", "-ac", "-rac", "-pl"):
            return False
        if any(s in al for s in NVIDIA_SMI_DENY_SUBSTR):
            return False
        if a.startswith("--query-") or a.startswith("--format"):
            i += 1
            continue
        if a in NVIDIA_SMI_ALLOW_EXACT or a.startswith("--list"):
            i += 1
            continue
        if a == "-d" or a.startswith("--display="):
            if a == "-d":
                if i + 1 >= len(rest):
                    return False
                section = rest[i + 1]
                i += 2
            else:
                section = a.split("=", 1)[1]
                i += 1
            if section.upper() not in NVIDIA_SMI_DISPLAY_SECTIONS:
                return False
            continue
        if a.startswith("-"):
            return False
        # positional leftovers not allowed
        return False
    return True


def clock_lock_int_text(value: int) -> str:
    """Return a digits-only form of a non-negative int, or raise."""
    if type(value) is not int or value < 0:
        raise ValueError("expected a non-negative integer")
    text = format(value, "d")
    if CLOCK_LOCK_INT.fullmatch(text) is None or int(text) != value:
        raise ValueError("expected a non-negative integer")
    return text


def nvidia_smi_is_clock_lock(argv: list[str] | tuple[str, ...]) -> bool:
    """True only for the two cap commands, with a validated GPU index and MHz.

    The only allowed shapes are::

        sudo -n nvidia-smi -i <gpu> -lgc 0,<MHz>
        sudo -n nvidia-smi -i <gpu> -rgc

    ``<gpu>`` and ``<MHz>`` are non-negative integers with no extra flags,
    no path to sudo or nvidia-smi, and no other sudo use.
    """
    parts = list(argv)
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        return False
    if parts[:4] != ["sudo", "-n", "nvidia-smi", "-i"]:
        return False
    if len(parts) == 6 and parts[5] == "-rgc":
        return CLOCK_LOCK_INT.fullmatch(parts[4]) is not None
    if len(parts) == 7 and parts[5] == "-lgc":
        spec = parts[6]
        if not spec.startswith("0,"):
            return False
        return (
            CLOCK_LOCK_INT.fullmatch(parts[4]) is not None
            and CLOCK_LOCK_INT.fullmatch(spec[2:]) is not None
        )
    return False


def observation_refusal(argv: list[str] | tuple[str, ...]) -> str | None:
    """Refuse anything that is not a node-fact read. None means the argv may run."""
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        return "refusing empty observation"
    prog = argv[0].rsplit("/", 1)[-1]
    rest = list(argv[1:])
    if prog == "sudo":
        return "refusing non-readonly command"
    if prog == "free" and rest in (["-b"], ["-m"], ["-g"]):
        return None
    if prog == "cat" and rest == ["/proc/meminfo"]:
        return None
    if prog == "cat" and len(rest) == 1 and rest[0].startswith("/") and ".." not in rest[0]:
        base = rest[0].rsplit("/", 1)[-1]
        if base in ("shadow", "passwd", "gshadow") or "dgx-announce" in rest[0]:
            return "refusing non-readonly command"
        return None
    if prog == "ss" and rest == ["-ltn"]:
        return None
    if prog == "nvidia-smi" and nvidia_smi_is_readonly(rest):
        return None
    if prog == "docker" and rest and rest[0] == "ps":
        if any(part in ("start", "stop", "restart", "rm", "kill") for part in rest):
            return "refusing non-readonly command"
        return None
    if prog == "docker" and rest and rest[0] == "inspect":
        if any(part in ("start", "stop", "restart", "rm", "kill") for part in rest):
            return "refusing non-readonly command"
        return None
    if prog == "docker" and rest[:2] == ["image", "inspect"] and len(rest) == 3 and not str(rest[2]).startswith("-"):
        return None
    if prog == "docker" and rest and rest[0] == "compose":
        banned = {"up", "down", "start", "stop", "restart", "rm", "kill", "exec"}
        if "config" in rest and not any(part in banned for part in rest):
            return None
    if prog == "systemctl" and rest and rest[0] in ("is-active", "show"):
        banned = {"start", "stop", "restart", "enable", "disable", "mask", "kill"}
        if any(part in banned for part in rest):
            return "refusing non-readonly command"
        return None
    if prog == "systemctl" and rest and rest[0] == "cat" and len(rest) == 2 and rest[1] not in ("start", "stop"):
        return None
    if prog == "ps" and rest in (["-eo", "args"], ["aux"], ["-ww", "-eo", "args"]):
        return None
    if prog == "test" and len(rest) == 2 and rest[0] in ("-e", "-f", "-d") and rest[1].startswith("/"):
        return None
    # Presence only. Stdout is the value and must be discarded by the caller.
    if prog == "printenv" and rest in (["HF_TOKEN"], ["HUGGING_FACE_HUB_TOKEN"]):
        return None
    return "refusing non-readonly command"


def subprocess_is_readonly(argv: list[str]) -> bool:
    """True if argv is an allowed observation command."""
    if not argv:
        return False
    prog = argv[0].rsplit("/", 1)[-1]
    if prog in FORBIDDEN_INVOKE or any(x in argv[0] for x in FORBIDDEN_INVOKE):
        return False
    rest = argv[1:]
    bad = {
        "start",
        "stop",
        "restart",
        "kill",
        "rm",
        "down",
        "up",
        "reload",
        "enable",
        "disable",
        "mask",
        "reset-failed",
        "isolate",
        "attach",
        "send-keys",
        "kill-session",
        "kill-server",
    }
    if prog == "systemctl":
        if any(a in bad for a in rest):
            return False
        return True
    if prog == "docker":
        if not rest:
            return False
        if rest[0] not in ("ps", "stats", "inspect"):
            return False
        if any(a in bad for a in rest):
            return False
        return True
    if prog == "nvidia-smi":
        return nvidia_smi_is_readonly(rest)
    if prog == "tmux":
        return False
    return False
