# SPDX-License-Identifier: Apache-2.0

"""Where a run goes: one Spark, or a set of Sparks, named by role.

A placement is a string. One role ("head", "worker", "node3", "node4") is a
single Spark. "both" is the head and worker together, the two-Spark form every
saved recipe and run record already uses. (A rendered launch row says "pair"
for any multi-Spark run; that word is a row kind, not a placement, so
``roles_of`` does not accept it.) Any other set is its roles joined
with "+" in [[nodes]] order, for example "worker+node3".

Every helper reads the configured roles, so a placement that names a Spark the
site does not have is simply not valid there.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

PAIR = "both"
LEGACY = ("head", "worker", PAIR)


def _roles(roles: Iterable[str] | None) -> tuple[str, ...]:
    if roles is None:
        from sparksi.config import CFG

        return CFG.roles
    return tuple(roles)


def roles_of(placement: str, roles: Iterable[str] | None = None) -> tuple[str, ...]:
    """Roles a placement covers, in node order; () when it names a Spark this site lacks."""
    known = _roles(roles)
    text = str(placement or "").strip()
    if text == PAIR:
        wanted = ("head", "worker")
    elif "+" in text:
        wanted = tuple(part.strip() for part in text.split("+"))
    else:
        wanted = (text,)
    if not wanted or len(set(wanted)) != len(wanted) or any(role not in known for role in wanted):
        return ()
    return tuple(role for role in known if role in wanted)


def placement_for(roles: Iterable[str], known: Iterable[str] | None = None) -> str:
    """The placement string for a set of roles (inverse of ``roles_of``); "" when empty or unknown."""
    order = _roles(known)
    chosen = set(roles)
    if not chosen or not chosen <= set(order):
        return ""
    ordered = [role for role in order if role in chosen]
    if len(ordered) == 1:
        return ordered[0]
    if ordered == ["head", "worker"]:
        return PAIR
    return "+".join(ordered)


def valid(placement: str, roles: Iterable[str] | None = None) -> bool:
    return bool(roles_of(placement, roles))


def size(placement: str, roles: Iterable[str] | None = None) -> int:
    return len(roles_of(placement, roles))


def is_multi(placement: str, roles: Iterable[str] | None = None) -> bool:
    return size(placement, roles) >= 2


def leader(placement: str, roles: Iterable[str] | None = None) -> str:
    """The role that leads a multi-Spark run (the first in node order); "" when not valid."""
    covered = roles_of(placement, roles)
    return covered[0] if covered else ""


def singles(roles: Iterable[str] | None = None) -> list[str]:
    return list(_roles(roles))


def multis(roles: Iterable[str] | None = None, *, max_size: int = 2) -> list[str]:
    """Every set of 2..max_size Sparks, smallest first, "both" first among the pairs."""
    known = _roles(roles)
    out: list[str] = []
    for count in range(2, min(max_size, len(known)) + 1):
        for combo in combinations(known, count):
            out.append(placement_for(combo, known))
    return out


def all_placements(roles: Iterable[str] | None = None, *, max_size: int = 2) -> list[str]:
    return singles(roles) + multis(roles, max_size=max_size)


def engine_view(placement: str, roles: Iterable[str] | None = None) -> tuple[str, dict[str, str]]:
    """How an engine builder sees a placement.

    Builders only know a lone "head", a lone "worker", and the "both" pair.
    The legacy placements pass through unchanged (so the two-Spark output is
    byte-for-byte what it was). Any other single Spark is built as "head", and
    any other pair as "both" (its first Spark leads). The map says which
    configured role each builder role stands for.
    """
    covered = roles_of(placement, roles)
    if placement in LEGACY and covered:
        return placement, {role: role for role in (covered if placement != PAIR else ("head", "worker"))}
    if len(covered) == 1:
        return "head", {"head": covered[0]}
    if len(covered) == 2:
        return PAIR, {"head": covered[0], "worker": covered[1]}
    return "", {}


def needs_text(placement: str, roles: Iterable[str] | None = None) -> str:
    """Plain words for why a placement cannot run on this site; "" when it can."""
    known = _roles(roles)
    if roles_of(placement, known):
        return ""
    text = str(placement or "").strip()
    if text in (PAIR, "pair"):
        wanted = ("head", "worker")
    elif "+" in text:
        wanted = tuple(part.strip() for part in text.split("+"))
    else:
        wanted = (text,)
    count = len(known)
    have = f"this site has {count} Spark" + ("" if count == 1 else "s")
    if len(wanted) >= 2 and len(wanted) > count:
        return f"needs {len(wanted)} Sparks; {have}"
    missing = [role for role in wanted if role not in known]
    if missing:
        return f"needs the {missing[0]} Spark; {have}"
    return f"not a placement Sparksi knows: {text or '(empty)'}"


def role_words(role: str) -> str:
    """Plain words for a role: "the head", "the worker", "Spark 3"."""
    text = str(role or "")
    if text in ("head", "worker"):
        return f"the {text}"
    if text.startswith("node") and text[4:].isdigit():
        return f"Spark {text[4:]}"
    return text or "the head"


def where_words(placement: str, roles: Iterable[str] | None = None) -> str:
    """Plain words for where a placement runs: "the head and the worker", "Spark 3"."""
    covered = roles_of(placement, roles) or ("head",)
    words = [role_words(role) for role in covered]
    if len(words) <= 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]
