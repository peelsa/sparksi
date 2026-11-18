# SPDX-License-Identifier: Apache-2.0

"""One place for every site-specific value: hosts, addresses, paths, ports.

Search order for the config file:

1. ``$SPARKSI_CONFIG``
2. ``$XDG_CONFIG_HOME/sparksi/sparksi.toml`` (default ``~/.config/sparksi/sparksi.toml``)
3. ``./config/sparksi.toml``

Nothing in the package hard-codes a site. With no config file the defaults
below describe a generic two-node example (``spark-a`` head, ``spark-b``
worker) using documentation addresses, which is also what the tests run on.

Environment overrides use ``SPARKSI_*``. The older ``DGX_MEM_*`` names are
still honored for one release (see ``env()``).
"""

from __future__ import annotations

import copy
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparksi import compat, protect

# Old env name -> new env name. Read only when the new name is unset.
LEGACY_ENV = {
    "SPARKSI_CATALOG": "DGX_MEM_CATALOG",
    "SPARKSI_STATE_DIR": "DGX_MEM_STATE_DIR",
    "SPARKSI_CONTROLLER_MODE": "DGX_MEM_CONTROLLER",
    "SPARKSI_NODE_ID": "DGX_ANNOUNCE_HOST",
    "SPARKSI_LOCK": "DGX_MEM_LOCK",
    "SPARKSI_SMI_TIMEOUT": "DGX_MEM_SMI_TIMEOUT",
    "SPARKSI_HTTP_TIMEOUT": "DGX_MEM_HTTP_TIMEOUT",
    "SPARKSI_CMD_TIMEOUT": "DGX_MEM_CMD_TIMEOUT",
}

APP_DIR = "sparksi"
LEGACY_APP_DIR = "dgx-mem"

DEFAULTS: dict[str, Any] = {
    "controller": {
        # Name of the machine that runs `sparksi serve` with the announce
        # listener. Matched against gethostname(), or forced with
        # SPARKSI_CONTROLLER_MODE=<name>.
        "name": "controller",
        "bind": ["127.0.0.1", "100.64.0.1"],
        "bind_deny": ["0.0.0.0", "::"],
        "access_log": "slow",  # "slow" logs requests over 1 s; "all" logs every request; "off".
        "monitor_upstream": "",  # Optional: GET /api/monitor from this controller (loopback or tailnet). Off when empty.
        "allowed_hosts": [],  # Extra names or addresses this controller answers to (beyond its own, found at startup).
        "ui_port": 7500,
        "announce_port": 7501,
        "announce_addr": "100.64.0.1",
        "reserved_ports": [8000, 8001, 8004, 8007, 8011],
        "state_dir": "",
        "catalog": "",
        "recipes_dir": "",
        "metrics_url": "http://127.0.0.1:8428",
        # Used memory of an idle node, when a status poll has not recorded one yet.
        "idle_baseline_gib": 10,
    },
    # One to four Sparks. The first is the head; the second, when present, is the
    # pair worker; a third and fourth are "node3" and "node4".
    "nodes": [
        {
            "id": "a",
            "name": "spark-a",
            "ssh": "spark-a",
            "tailscale": "100.64.0.11",
            "bind_deny": ["192.0.2.11", "198.51.100.11"],
            "uma_gib": 121,
            "fabric": {"interface": "enp1s0f1np1", "ip": "192.0.2.11", "hcas": ["roce0", "roce1"]},
            "mount_roots": ["/srv/example/cache", "/srv/example/builds"],
        },
        {
            "id": "b",
            "name": "spark-b",
            "ssh": "spark-b",
            "tailscale": "100.64.0.12",
            "bind_deny": [],
            "uma_gib": 121,
            "fabric": {"interface": "enp1s0f1np1", "ip": "192.0.2.12", "hcas": ["roce0", "roce1"]},
            "mount_roots": ["/srv/example/cache", "/srv/example/worker"],
        },
    ],
    # Exact argv lists the node agent may run for a catalog service.
    "service_commands": [
        ["/srv/example/web/start.sh"],
        ["/srv/example/web/stop.sh"],
    ],
    # Opt-in dashboard control for one protected service. Off unless a site file
    # sets user_controllable and both commands. The table name is the catalog id.
    "services": {},
    # Pinned engine images. Tags are never "latest". A saved recipe may only
    # name an image from allowed_images.
    "engines": {
        "vllm_image": "nvcr.io/nvidia/vllm:26.05-py3",
        "sglang_image": "nvcr.io/nvidia/sglang:26.02-py3",
        "trtllm_image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6",
        "allowed_images": [
            "nvcr.io/nvidia/vllm:26.05-py3",
            "nvcr.io/nvidia/sglang:26.02-py3",
            "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6",
        ],
        "vllm_master_port": 29501,
        "sglang_dist_port": 50000,
        "kv_reserve_gib": 8,
    },
    "archive": {
        "id": "archive",
        "node": "a",
        "root": "/mnt/archive",
        "label": "Archive (RAID 1 DAS)",
        # Extra roots that must never be written or deleted (for example the
        # raw mount that the archive is bind-mounted from).
        "extra_blocked": ["/mnt/archive-raw"],
    },
    "paths": {
        "home": "/home/USER",
        "hf_hub": "/home/USER/.cache/huggingface/hub",
        "video_service_root": "/home/USER/services/video",
        "model_inventory": "/home/USER/ops/model-inventory.yaml",
        "disk_inventory": "/home/USER/scripts/dgx_models.py",
        "blocked_prefixes": ["/home/USER-agent"],
        "too_broad": ["/home/USER/.cache", "/home/USER/services"],
        # Prefixes of the direct node-to-node link (shown as "direct" IPs).
        "direct_prefixes": ["198.18.0.", "198.18.1."],
        # Prefixes that mark a LAN endpoint in tailscale status.
        "lan_prefixes": [],  # empty: any private, non-CGNAT IPv4
    },
    "extra_machines": {
        # Tailscale HostName -> id shown in the Machines panel.
        "tailscale": {},
    },
    "compat": {
        # Older API readers can be served the previous archive id and mount
        # key. Leave empty in new installs.
        "legacy_archive_id": "",
        "legacy_mount_key": "",
        # Older catalog row keys -> current names, renamed when the catalog loads.
        "legacy_row_keys": {},
    },
}


def env(name: str, default: str | None = None) -> str | None:
    """Read SPARKSI_* first, then its deprecated alias."""
    val = os.environ.get(name)
    if val not in (None, ""):
        return val
    old = LEGACY_ENV.get(name)
    if old:
        val = os.environ.get(old)
        if val not in (None, ""):
            return val
    val = compat.legacy_env(name)
    if val is not None:
        return val
    return default


def state_subdir(base: Path) -> Path:
    """``base/sparksi``, or the pre-rename ``base/dgx-mem`` if only that exists
    (so a cutover keeps the lock, ledgers and unlock token in place)."""
    new, old = Path(base) / APP_DIR, Path(base) / LEGACY_APP_DIR
    if old.is_dir() and not new.exists():
        return old
    return new


def config_path() -> Path | None:
    explicit = os.environ.get("SPARKSI_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    for cand in (xdg / "sparksi" / "sparksi.toml", Path.cwd() / "config" / "sparksi.toml"):
        if cand.is_file():
            return cand
    return None


def _fabric(raw: dict[str, Any] | None) -> Fabric:
    raw = raw or {}
    hcas = raw.get("hcas") or []
    if isinstance(hcas, str):
        hcas = [part.strip() for part in hcas.split(",") if part.strip()]
    return Fabric(
        interface=str(raw.get("interface") or ""),
        ip=str(raw.get("ip") or ""),
        hcas=[str(item) for item in hcas if str(item)],
    )


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


MAX_NODES = 4


class SingleSparkError(RuntimeError):
    """Raised when code asks for the worker Spark on a one-Spark site."""


def role_name(index: int) -> str:
    """Role word for the Spark at this position in [[nodes]]."""
    if index == 0:
        return "head"
    if index == 1:
        return "worker"
    return f"node{index + 1}"


ALL_ROLES = tuple(role_name(index) for index in range(MAX_NODES))


@dataclass
class Fabric:
    """Node-to-node link. Read from config; launch code never writes it back."""

    interface: str = ""
    ip: str = ""
    hcas: list[str] = field(default_factory=list)


@dataclass
class Node:
    id: str
    name: str
    ssh: str
    tailscale: str = ""
    bind_deny: list[str] = field(default_factory=list)
    uma_gib: float = 121
    fabric: Fabric = field(default_factory=Fabric)
    mount_roots: list[str] = field(default_factory=list)


def _check_machines(nodes: list[Any], endpoints: list[Any]) -> None:
    """Refuse a hand-edited bad value before it can reach ssh, a URL or the page."""
    from urllib.parse import urlsplit

    from sparksi.hostcheck import check_host, check_id, check_name, check_port, check_ssh

    def fail(where: str, key: str, exc: ValueError) -> None:
        raise ValueError(f"sparksi.toml {where} {key}: {exc}") from None

    for i, n in enumerate(nodes, 1):
        where = f"[[nodes]] #{i}"
        if not isinstance(n, dict) or "id" not in n:
            raise ValueError(f"sparksi.toml {where}: needs an id")
        # The ssh check runs on the destination Sparksi will actually use (ssh, else name, else id).
        checks = [("id", check_id, str(n["id"])), ("ssh", check_ssh, n.get("ssh") or n.get("name") or str(n["id"]))]
        if "name" in n:
            checks.append(("name", check_name, n["name"]))
        for key, check, value in checks:
            try:
                check(value)
            except ValueError as exc:
                fail(where, key, exc)
        if n.get("tailscale"):
            try:
                check_host(n["tailscale"])
            except ValueError as exc:
                fail(where, "tailscale", exc)
    for i, e in enumerate(endpoints, 1):
        where = f"[[endpoints]] #{i}"
        if not isinstance(e, dict):
            raise ValueError(f"sparksi.toml {where}: must be a table")
        try:
            check_id(e.get("id"))
        except ValueError as exc:
            fail(where, "id", exc)
        try:
            check_name(e.get("name"))
        except ValueError as exc:
            fail(where, "name", exc)
        parts = urlsplit(str(e.get("url") or ""))
        try:
            if parts.scheme != "http" or parts.username or parts.password or parts.path not in ("", "/") or parts.query or parts.fragment:
                raise ValueError("must be http://host:port")
            check_host(parts.hostname or "")
            check_port(parts.port)
        except ValueError as exc:
            fail(where, "url", exc)


class Config:
    def __init__(self, raw: dict[str, Any] | None = None, source: Path | None = None) -> None:
        self.file_raw = dict(raw or {})  # as written; sparksi.protect reads legacy markers here
        self.raw = _merge(DEFAULTS, compat.apply_legacy_paths(raw))
        self.source = source
        nodes = self.raw.get("nodes")
        if nodes is None:
            nodes = DEFAULTS["nodes"]
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("sparksi.toml needs at least one [[nodes]] table (the head Spark)")
        if len(nodes) > MAX_NODES:
            raise ValueError(
                f"sparksi.toml lists {len(nodes)} [[nodes]]; Sparksi manages 1 to {MAX_NODES} Sparks"
            )
        _check_machines(nodes, self.raw.get("endpoints") or [])
        seen: set[str] = set()
        for i, n in enumerate(nodes, 1):
            key = str(n["id"])
            if key in seen:
                raise ValueError(f"sparksi.toml [[nodes]] #{i} id: {key!r} is used by an earlier Spark")
            seen.add(key)
        self.nodes = [
            Node(
                id=str(n["id"]),
                name=str(n.get("name") or n["id"]),
                ssh=str(n.get("ssh") or n.get("name") or n["id"]),
                tailscale=str(n.get("tailscale") or ""),
                bind_deny=list(n.get("bind_deny") or []),
                uma_gib=float(n.get("uma_gib") or 121),
                fabric=_fabric(n.get("fabric") if isinstance(n.get("fabric"), dict) else None),
                mount_roots=[str(item) for item in (n.get("mount_roots") or []) if str(item)],
            )
            for n in nodes
        ]

    # --- nodes -----------------------------------------------------------
    @property
    def head(self) -> Node:
        return self.nodes[0]

    @property
    def worker(self) -> Node:
        """The second Spark. A one-Spark site has none: ask ``has_worker`` first."""
        if len(self.nodes) < 2:
            raise SingleSparkError("this needs a second Spark, and sparksi.toml lists one")
        return self.nodes[1]

    @property
    def has_worker(self) -> bool:
        return len(self.nodes) >= 2

    @property
    def roles(self) -> tuple[str, ...]:
        """Role of each configured Spark, in file order: head, worker, node3, node4."""
        return tuple(role_name(index) for index in range(len(self.nodes)))

    def role_nodes(self) -> list[tuple[str, Node]]:
        return [(role_name(index), node) for index, node in enumerate(self.nodes)]

    def node_for_role(self, role: str) -> Node | None:
        """The Spark that holds ``role``, or None when this site has no such Spark."""
        try:
            return self.nodes[self.roles.index(str(role))]
        except ValueError:
            return None

    @property
    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]

    @property
    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def name_for(self, node_id: str) -> str:
        for n in self.nodes:
            if n.id == node_id:
                return n.name
        return node_id

    def id_for(self, name: str) -> str:
        for n in self.nodes:
            if n.name == name:
                return n.id
        return name

    def node_by_role(self, role: str) -> Node:
        """The Spark for ``role``; an unknown role (or a missing worker) reads as the head."""
        return self.node_for_role(role) or self.head

    def role_for(self, name: str) -> str:
        """Role of the Spark with this name, id or role word; anything else reads as head."""
        for role, node in self.role_nodes():
            if name in (node.name, node.id, role):
                return role
        return "head"

    def announce_allow(self) -> dict[str, str]:
        """Source tailnet address -> node id, for the announce listener."""
        return {n.tailscale: n.id for n in self.nodes if n.tailscale}

    # --- controller ------------------------------------------------------
    @property
    def controller(self) -> dict[str, Any]:
        return self.raw["controller"]

    @property
    def controller_name(self) -> str:
        return str(self.controller.get("name") or "controller")

    def node_bind_deny(self) -> list[str]:
        out: list[str] = ["0.0.0.0", "::"]
        for n in self.nodes:
            out.extend(n.bind_deny)
        return out

    # --- archive / paths -------------------------------------------------
    @property
    def archive(self) -> dict[str, Any]:
        return self.raw["archive"]

    @property
    def archive_id(self) -> str:
        return str(self.archive.get("id") or "archive")

    @property
    def archive_node(self) -> str:
        """Node *name* that owns the archive store."""
        return self.name_for(str(self.archive.get("node") or self.head.id))

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    def path(self, key: str) -> str:
        return str(self.paths.get(key) or "")

    @property
    def engines(self) -> dict[str, Any]:
        raw = self.raw.get("engines")
        return raw if isinstance(raw, dict) else {}

    def engine_image(self, engine: str) -> str:
        return str(self.engines.get(f"{engine}_image") or "")

    def allowed_images(self) -> list[str]:
        raw = self.engines.get("allowed_images") or []
        if isinstance(raw, str):
            raw = [raw]
        return [str(item) for item in raw if str(item) and ":latest" not in str(item)]

    def hf_cache(self) -> str:
        """Directory mounted at ``/root/.cache/huggingface`` inside engine containers."""
        hub = self.path("hf_hub") or "/home/USER/.cache/huggingface/hub"
        path = Path(hub)
        if path.name == "hub":
            return str(path.parent)
        return hub or "/home/USER/.cache/huggingface"

    def mount_roots(self) -> list[str]:
        """Host paths a generated recipe may bind into a container.

        The recipe generator mounts ``hf_cache`` and nothing above it.
        """
        roots: list[str] = []
        for item in (self.hf_cache(), self.path("hf_hub")):
            text = str(item or "")
            if text and text not in roots:
                roots.append(text)
        for node in self.nodes:
            for item in node.mount_roots:
                text = str(item or "")
                if text and text not in roots:
                    roots.append(text)
        return roots

    def service_commands(self) -> list[list[str]]:
        """Exact argv lists a catalog service may run. Matched verbatim."""
        raw = self.raw.get("service_commands") or []
        found: list[list[str]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, list) and item and all(isinstance(part, str) and part for part in item):
                    found.append(list(item))
        for spec in self._controllable_services().values():
            for key in ("start", "stop"):
                argv = list(spec[key])
                if argv not in found:
                    found.append(argv)
        return found

    def _service_argv(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(part) for part in value if str(part)]

    def _controllable_services(self) -> dict[str, dict[str, Any]]:
        """Protected services the dashboard may start and stop. Empty unless opted in."""
        raw = self.raw.get("services") or {}
        found: dict[str, dict[str, Any]] = {}
        if not isinstance(raw, dict):
            return found
        for key, spec in raw.items():
            if not isinstance(spec, dict) or not spec.get("user_controllable"):
                continue
            start = self._service_argv(spec.get("start"))
            stop = self._service_argv(spec.get("stop"))
            if not start or not stop:
                continue
            try:
                reserve = float(spec.get("reserve_gib") if spec.get("reserve_gib") not in (None, "") else 40)
            except (TypeError, ValueError):
                reserve = 40.0
            found[str(key)] = {"start": start, "stop": stop, "reserve_gib": reserve}
        return found

    def controllable_service(self, service_id: str) -> dict[str, Any] | None:
        """Start and stop argv for one opted-in service, or None."""
        return self._controllable_services().get(str(service_id or ""))

    def kv_reserve_gib(self) -> float:
        try:
            return float(self.engines.get("kv_reserve_gib") or 8)
        except (TypeError, ValueError):
            return 8.0

    def master_port(self) -> int:
        try:
            return int(self.engines.get("vllm_master_port") or 29501)
        except (TypeError, ValueError):
            return 29501

    def sglang_dist_port(self) -> int:
        try:
            return int(self.engines.get("sglang_dist_port") or 50000)
        except (TypeError, ValueError):
            return 50000

    @property
    def compat(self) -> dict[str, Any]:
        return self.raw["compat"]

    def agent_config(self) -> dict[str, Any]:
        """Values the stdlib-only node agent needs; sent with each request."""
        blocked = [str(self.archive.get("root") or "/mnt/archive")]
        blocked += [str(p) for p in self.archive.get("extra_blocked") or []]
        blocked += [str(p) for p in self.paths.get("blocked_prefixes") or []]
        if self.path("video_service_root"):
            blocked.append(self.path("video_service_root"))
        home = self.path("home")
        return {
            "archive_root": str(self.archive.get("root") or "/mnt/archive"),
            "blocked_prefixes": blocked,
            "hub_root": self.path("hf_hub"),
            "too_broad": ["/", "/home", home, *[str(p) for p in self.paths.get("too_broad") or []]],
            "direct_prefixes": [str(p) for p in self.paths.get("direct_prefixes") or []],
            "mount_roots": self.mount_roots(),
            "service_commands": self.service_commands(),
            # Off unless the file sets [controller] stop_unmanaged = true; the agent refuses otherwise.
            "stop_unmanaged": self.controller.get("stop_unmanaged") is True,
            **protect.agent_values(),
        }

    def effective(self) -> dict[str, Any]:
        """The settings in force: built-in defaults merged with the file, secrets redacted.

        Shown by ``sparksi config show``. A copy; the live settings are not shared.
        """
        return redact(copy.deepcopy(self.raw))

    def public(self) -> dict[str, Any]:
        """Display values for ``GET /api/config``.

        Ids, names and labels only. Addresses, SSH targets, paths, bind and
        deny lists, and compat key names are not copied out of the raw file.
        """
        nodes = [{"role": role, "id": node.id, "name": node.name} for role, node in self.role_nodes()]
        out: dict[str, Any] = {
            "nodes": nodes,
            "head": {"id": self.head.id, "name": self.head.name},
        }
        if self.has_worker:
            out["worker"] = {"id": self.worker.id, "name": self.worker.name}
        out.update({
            "controller": {"name": self.controller_name},
            "archive": {
                "id": self.archive_id,
                "label": str(self.archive.get("label") or "Archive"),
            },
        })
        return out


# Top-level tables the rest of the code reads with ``.get``. A file that makes one of
# them a string or list would fail later, far from the file, so it is refused at load.
_TABLES = ("controller", "engines", "archive", "paths", "extra_machines", "compat", "protect", "services")

_SECRET_KEY = re.compile(r"(token|secret|password|passwd|credentials?|api_?key|private_key)$", re.IGNORECASE)
_URL_USER = re.compile(r"(?<=://)[^/@\s]+@")
REDACTED = "<redacted>"


def redact(value: Any, key: str = "") -> Any:
    """A copy of a settings value with secrets hidden.

    A value under a key that ends in token, secret, password, passwd, credential(s),
    api_key or private_key is replaced when it is not empty. User info in a URL is
    dropped. No key in the shipped settings holds a secret; this covers a site that
    adds one.
    """
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if value in ("", None, False) or isinstance(value, (int, float)):
        return value
    if _SECRET_KEY.search(key):
        return REDACTED
    if isinstance(value, str):
        return _URL_USER.sub(REDACTED + "@", value)
    return value


def _shape_problems(raw: dict[str, Any]) -> list[str]:
    return [
        f"sparksi.toml [{name}] must be a table"
        for name in _TABLES
        if name in raw and not isinstance(raw[name], dict)
    ]


def load(path: Path | None = None) -> Config:
    path = path or config_path()
    if path is None:
        return Config()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    bad = _shape_problems(raw)
    if bad:
        raise ValueError("; ".join(bad))
    return Config(raw, source=path)


def check(path: Path | None = None) -> tuple[Config | None, list[str]]:
    """Load the site file without raising: ``(config, [])``, or ``(None, messages)``.

    Without a path the usual search order applies, and no file is not a problem.
    Every way a file can be unusable (unreadable, not TOML, not UTF-8, a bad table
    or node) comes back as a message that names the file.
    """
    target = path or config_path()
    try:
        return load(target), []
    except OSError as exc:
        return None, [f"cannot read {target}: {exc.strerror or exc}"]
    except KeyError as exc:
        return None, [f"{target}: missing key {exc.args[0]!r}"]
    except ValueError as exc:  # includes tomllib.TOMLDecodeError and UnicodeDecodeError
        text = str(exc)
        # Load messages open with the generic file name; say which file it is.
        text = f"{target}{text[len('sparksi.toml'):]}" if text.startswith("sparksi.toml ") else f"{target}: {text}"
        return None, [text]
    except (TypeError, AttributeError) as exc:
        return None, [f"{target}: a value has the wrong type ({exc})"]


def _bootstrap() -> tuple[Config, list[str]]:
    """Load the site file at import. An invalid file must not raise here.

    The built-in example config is kept so every import succeeds, and one warning
    lists the problems. The CLI and the monitor read ``CFG_ERRORS`` and refuse to
    run, so nothing acts on the placeholder nodes.
    """
    cfg, errors = check()
    if cfg is not None:
        return cfg, []
    from sparksi.log import logger

    logger().warning("ignoring invalid config: %s", "; ".join(errors))
    return Config(), errors


_boot, _boot_errors = _bootstrap()
CFG = _boot
# Problems found when the site file was loaded at import; empty when it is valid.
CFG_ERRORS: list[str] = list(_boot_errors)


def apply_agent(cfg: Config | None = None) -> None:
    """Push the loaded site file into the in-process agent.

    Mount roots come from ``agent_config`` (the cache plus each node's
    ``mount_roots``), the same paths mount validation reads. The agent's
    baked-in placeholder list is replaced, including when the file sets none.
    """
    from sparksi.agent import configure

    configure((cfg or CFG).agent_config())


def reload(path: Path | None = None) -> Config:
    """Re-read config (tests, or after editing the file). Modules that copied
    values at import time keep their copies; prefer reading ``CFG`` at call time.

    The in-process agent is updated from the file just loaded, so a plan or
    launch after a reload sees the configured mount roots.
    """
    fresh = load(path)
    CFG.__init__(fresh.raw, source=fresh.source)
    CFG_ERRORS.clear()
    apply_agent(CFG)
    return CFG
