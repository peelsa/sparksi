#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build a public export of a Sparksi ref: a one-commit root, or one commit on the public main.

The export is a new local repository at ``--out``. This tool does not push,
does not create a remote repository, and does not change remote settings.
It prints the push command for a person to run later.

Author name and email are required inputs. The Peelsa machine user is not
chosen in this script.

Local rehearsal (writes the export directory and a sibling report; does not push):

    python3 scripts/export_public.py \\
        --ref HEAD \\
        --out /tmp/sparksi-public \\
        --author-name "<Peelsa machine account>" \\
        --author-email "noreply@example.com" \\
        --version 0.7.0 \\
        --add-spdx \\
        --allow-no-metadata-check

``--version`` must match ``[project].version`` in that ref's ``pyproject.toml``.
The report is ``/tmp/EXPORT_REPORT.md``, beside the export directory, and is not
part of the release commit. Replace the author values with the machine
user once it exists. Pass ``--add-spdx`` when the source tree has no SPDX
headers yet; headers are inserted into the export only. Pass
``--allow-no-metadata-check`` only while ``check_identifiers.py`` has no
``--git-metadata`` flag. The final run omits that flag and passes
``--require-gitleaks``. Delete the output directory and the sibling report
when finished.

Incremental export (one new commit on top of the current public main; no tag):

    git clone <public-url> /tmp/public-main          # read-only clone
    SPARKSI_ALLOWED_AUTHORS=<author-email> python3 scripts/export_public.py \\
        --ref <internal-ref> \\
        --out /tmp/sparksi-public-sync \\
        --author-name "<Peelsa machine account>" \\
        --author-email "<author-email>" \\
        --version 0.6.0 \\
        --parent-repo /tmp/public-main --parent-ref main \\
        --expect-parent <public main sha> \\
        --branch sync/0.6.0 --require-gitleaks

The new commit's parent is the fetched public main. The export tree is the
source ref minus the exclude list, plus the public-only files kept under
``public/`` in the source, with each workflow branch filter set to ``main``.
Push the printed branch and open a pull request against public main.

``--author-email`` must be listed in ``SPARKSI_ALLOWED_AUTHORS``. That
variable is one address per line, or comma-separated. The match ignores
case. An empty value allows no author. The release flags do not fill it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path

SPDX_TEXT = "SPDX-License-Identifier: Apache-2.0"
SPDX_LINE = f"# {SPDX_TEXT}\n"
REPORT_NAME = "EXPORT_REPORT.md"
STATIC_PREFIX = "src/sparksi/static/"
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
ALLOWED_GIT = frozenset({
    "add",
    "archive",
    "cat-file",
    "commit",
    "commit-tree",
    "config",
    "fetch",  # only from a local --parent-repo directory; see make_incremental
    "init",
    "log",
    "ls-tree",
    "rev-parse",
    "tag",
    "update-ref",
    "write-tree",
})
PUBLIC_LAYER = "public"
WORKFLOW_DIR = ".github/workflows"
PUBLIC_BRANCH = "main"
CHECKERS = ("check_identifiers.py", "check_ui_colors.py", "check_stdlib.py")
_CHECK_TEXT = None


class ExportError(Exception):
    """The export cannot be built. No commit is published."""


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclass
class Outcome:
    source_ref: str
    version: str
    author_name: str
    author_email: str
    commit_date: str
    file_count: int
    tree_sha256: str
    excluded: list[str]
    checks: list[CheckResult] = field(default_factory=list)
    commit_sha: str = ""
    git_tree: str = ""
    public_url: str = "<public-url>"
    denylist_used: bool = False
    parent_sha: str = ""
    branch: str = ""
    message: str = ""
    public_layer: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(item.failed for item in self.checks)


def push_command(url: str, branch: str = "") -> str:
    if branch:
        return f"git push {url} main:refs/heads/{branch}"
    return f"git push --set-upstream {url} main --tags"


def git_subcommand(args: list[str]) -> str:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-C", "-c", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg
    return ""


def git_run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run one local git command. Remote and publish commands are refused."""
    command = git_subcommand(args)
    if command not in ALLOWED_GIT:
        raise ExportError(f"refusing to run git {command or '(no command)'}")
    run_env = os.environ.copy() if env is None else dict(env)
    run_env["GIT_NO_LAZY_FETCH"] = "1"
    try:
        proc = subprocess.run(
            ["git", *args],
            env=run_env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ExportError("git is not on PATH") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise ExportError(detail or f"git {command} exited {proc.returncode}")
    return proc


def git_text(args: list[str], *, env: dict[str, str] | None = None) -> str:
    return git_run(args, env=env).stdout.decode("utf-8", "replace")


def resolve_ref(repo: Path, ref: str) -> str:
    sha = git_text(["-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"]).strip()
    if not sha:
        raise ExportError(f"could not resolve ref {ref}")
    return sha


def ref_commit_date(repo: Path, rev: str) -> str:
    """Committer date of the source ref, as strict ISO 8601."""
    date = git_text(["-C", str(repo), "log", "-1", "--format=%cI", rev]).strip()
    if not date or any(ch.isspace() for ch in date):
        raise ExportError("ref commit date is not a single value")
    return date


def to_utc_git_date(value: str) -> str:
    """Normalize a commit date to UTC with a ``+0000`` suffix."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExportError("ref commit date is not a single value") from exc
    if parsed.tzinfo is None:
        raise ExportError("ref commit date has no timezone")
    utc = parsed.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%d %H:%M:%S +0000")


def archive_tree(repo: Path, rev: str, dest: Path) -> None:
    data = git_run(["-C", str(repo), "archive", "--format=tar", rev]).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        archive.extractall(dest, filter="data")


def load_patterns(path: Path) -> list[str]:
    if not path.is_file():
        raise ExportError(f"exclude list not found: {path}")
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def pattern_matches(rel: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        prefix = pattern[:-1]
        return rel == prefix or rel.startswith(prefix + "/")
    name = rel.rsplit("/", 1)[-1]
    return fnmatchcase(rel, pattern) or fnmatchcase(name, pattern)


def apply_excludes(root: Path, patterns: list[str]) -> list[str]:
    removed: list[str] = []
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel == prior or rel.startswith(prior + "/") for prior in removed):
            continue
        if not any(pattern_matches(rel, pattern) for pattern in patterns):
            continue
        removed.append(rel)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return removed


def needs_spdx(path: Path) -> bool:
    """Python, shell, and extensionless shebang scripts carry an SPDX header."""
    if not path.is_file():
        return False
    if path.suffix in {".py", ".sh"}:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as handle:
            prefix = handle.read(2)
    except OSError:
        return False
    return prefix == b"#!"


def source_files(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if ".git" in path.relative_to(root).parts or not needs_spdx(path):
            continue
        found.append(path)
    found.sort(key=lambda item: item.relative_to(root).as_posix())
    return found


def header_has_spdx(text: str) -> bool:
    """True when one of the first three lines already carries the SPDX header."""
    return any(SPDX_TEXT in line for line in text.splitlines()[:3])


def insert_spdx(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if header_has_spdx(text):
        return False
    lines = text.splitlines(keepends=True)
    index = 1 if lines and lines[0].startswith("#!") else 0
    block = SPDX_LINE
    if index >= len(lines) or lines[index].strip():
        block += "\n"
    if not lines:
        path.write_text(SPDX_LINE, encoding="utf-8")
    else:
        lines.insert(index, block)
        path.write_text("".join(lines), encoding="utf-8")
    return True


def add_spdx_headers(root: Path) -> list[str]:
    inserted = []
    for path in source_files(root):
        try:
            changed = insert_spdx(path)
        except (UnicodeDecodeError, OSError) as exc:
            raise ExportError(f"could not read {path.relative_to(root).as_posix()}: {exc}") from exc
        if changed:
            inserted.append(path.relative_to(root).as_posix())
    return inserted


def spdx_check(root: Path, inserted: list[str]) -> CheckResult:
    missing = []
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            missing.append(f"{rel}: {exc}")
            continue
        if not header_has_spdx(text):
            missing.append(rel)
    detail = "every .py, .sh, and extensionless script has an SPDX header in its first three lines"
    if inserted:
        detail += "\nheaders inserted into the export only:\n" + "\n".join(inserted)
    if missing:
        return CheckResult("spdx", "fail", "missing SPDX-License-Identifier: Apache-2.0\n" + "\n".join(missing))
    return CheckResult("spdx", "pass", detail)


def is_logo(path: str) -> bool:
    if not path.startswith(STATIC_PREFIX):
        return False
    name = path.rsplit("/", 1)[-1]
    if name == "peelsa.png":
        return True
    return name.startswith("sparksi-wordmark-transparent") and name.endswith(".png")


def logo_paths(repo: Path, rev: str) -> list[str]:
    listing = git_text(["-C", str(repo), "ls-tree", "-r", "--name-only", rev])
    return sorted(line for line in listing.splitlines() if is_logo(line))


def verify_logos(repo: Path, rev: str, root: Path) -> CheckResult:
    paths = logo_paths(repo, rev)
    names = [path.rsplit("/", 1)[-1] for path in paths]
    problems = []
    if "peelsa.png" not in names:
        problems.append("source ref has no src/sparksi/static/peelsa.png")
    if not any(name.startswith("sparksi-wordmark-transparent") for name in names):
        problems.append("source ref has no SparkSI wordmark under src/sparksi/static/")
    lines = []
    for rel in paths:
        blob = git_run(["-C", str(repo), "cat-file", "blob", f"{rev}:{rel}"]).stdout
        exported = root / rel
        source_hash = hashlib.sha256(blob).hexdigest()
        if not exported.is_file():
            problems.append(f"{rel}: missing from the export")
            lines.append(f"{rel}: source {source_hash} export missing")
            continue
        export_hash = hashlib.sha256(exported.read_bytes()).hexdigest()
        lines.append(f"{rel}: {export_hash}")
        if export_hash != source_hash or exported.read_bytes() != blob:
            problems.append(f"{rel}: bytes differ from the source ref")
    detail = "\n".join(lines)
    if problems:
        return CheckResult("logos", "fail", "\n".join(problems + lines))
    return CheckResult("logos", "pass", detail or "no logo bytes to compare")


def export_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".git" in rel.parts or rel.as_posix() == REPORT_NAME:
            continue
        if path.is_symlink() or path.is_file():
            files.append(path)
    files.sort(key=lambda item: item.relative_to(root).as_posix())
    return files


def tree_sha256(root: Path) -> str:
    """SHA-256 of sorted ``<file sha256><two spaces><path>`` lines.

    ``.git`` is omitted. Each line ends with a newline.
    """
    lines = []
    for path in export_files(root):
        rel = path.relative_to(root).as_posix()
        payload = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        lines.append(f"{digest}  {rel}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def clip(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def run_check(name: str, cmd: list[str], cwd: Path) -> CheckResult:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    detail = clip(proc.stdout + (("\n" + proc.stderr) if proc.stderr else ""))
    status = "pass" if proc.returncode == 0 else "fail"
    if not detail:
        detail = "clean" if status == "pass" else f"exited {proc.returncode}"
    return CheckResult(name, status, detail)


def checker_supports_git_metadata(script: Path) -> bool:
    try:
        return "--git-metadata" in script.read_text(encoding="utf-8")
    except OSError:
        return False


def run_gitleaks(root: Path, *, required: bool) -> CheckResult:
    if shutil.which("gitleaks") is None:
        if required:
            return CheckResult("gitleaks", "fail", "gitleaks is not on PATH; --require-gitleaks is set")
        return CheckResult("gitleaks", "warn", "gitleaks is not on PATH; secret scan skipped")
    proc = subprocess.run(
        ["gitleaks", "detect", "--source", str(root), "--no-git", "--redact"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = clip(proc.stdout + (("\n" + proc.stderr) if proc.stderr else ""))
    if proc.returncode == 0:
        return CheckResult("gitleaks", "pass", detail or "no leaks found")
    return CheckResult("gitleaks", "fail", detail or f"gitleaks exited {proc.returncode}")


def release_env(name: str, email: str, when: str, blank_config: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_GLOBAL": str(blank_config),
        "GIT_CONFIG_SYSTEM": str(blank_config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_AUTHOR_DATE": when,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
        "GIT_COMMITTER_DATE": when,
        "GIT_EDITOR": "false",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return env


def identity_args(name: str, email: str) -> list[str]:
    return ["-c", f"user.name={name}", "-c", f"user.email={email}", "-c", "commit.gpgsign=false"]


def release_message(version: str) -> str:
    return f"Sparksi {version}: initial public release"


def tag_message(version: str) -> str:
    return f"Sparksi {version}"


def identity_denylist() -> list[str]:
    raw = os.environ.get("SPARKSI_IDENTITY_DENYLIST", "")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def allowed_author_emails() -> set[str]:
    """Addresses in ``SPARKSI_ALLOWED_AUTHORS``. The release flag does not fill this."""
    raw = os.environ.get("SPARKSI_ALLOWED_AUTHORS", "")
    found: set[str] = set()
    for chunk in raw.replace(",", "\n").splitlines():
        item = chunk.strip().lower()
        if item:
            found.add(item)
    return found


def release_check_text():
    """Load ``check_text`` from the guard beside this script."""
    global _CHECK_TEXT
    if _CHECK_TEXT is not None:
        return _CHECK_TEXT
    path = Path(__file__).resolve().parent / "check_identifiers.py"
    spec = importlib.util.spec_from_file_location("sparksi_export_identifiers", path)
    if spec is None or spec.loader is None:
        raise ExportError("could not load check_identifiers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    function = getattr(module, "check_text", None)
    if function is None:
        raise ExportError("check_identifiers.py has no check_text")
    _CHECK_TEXT = function
    return function


def sync_message(version: str) -> str:
    return f"Sparksi {version}: sync from the source tree"


def check_release_identity(name: str, email: str, version: str, message: str = "") -> CheckResult:
    """Fail when the release identity or messages trip the identity guard."""
    check_text = release_check_text()
    extra = identity_denylist()
    fields = {
        "author name": name,
        "author email": email,
        "commit message": message or release_message(version),
        "tag message": tag_message(version),
    }
    hits = []
    for label, text in fields.items():
        for hit in check_text(text, extra):
            hits.append(f"{label}: {hit}")
    if hits:
        return CheckResult("release identity", "fail", "\n".join(hits))
    return CheckResult(
        "release identity",
        "pass",
        "author name, author email, commit message, and tag message are clean",
    )


def project_version(path: Path) -> str:
    if not path.is_file():
        raise ExportError("pyproject.toml is missing from the export")
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            continue
        if current != "project":
            continue
        match = re.match(r'version\s*=\s*"([^"]+)"', line)
        if match:
            return match.group(1)
    raise ExportError("pyproject.toml has no project version")


def version_check(root: Path, version: str) -> CheckResult:
    try:
        found = project_version(root / "pyproject.toml")
    except (ExportError, UnicodeDecodeError, OSError) as exc:
        return CheckResult("version", "fail", str(exc))
    if found != version:
        return CheckResult("version", "fail", f"pyproject.toml version is {found}; --version is {version}")
    return CheckResult("version", "pass", f"pyproject.toml version {found}")


def strip_coauthored(message: str) -> str:
    kept = [line for line in message.splitlines() if not line.strip().lower().startswith("co-authored-by:")]
    return "\n".join(kept).strip()


def scrub_git_dir(work: Path) -> None:
    """Drop reflogs and the commit-message scratch file from the export."""
    git_dir = work / ".git"
    if not git_dir.is_dir():
        return
    logs = git_dir / "logs"
    if logs.is_dir():
        shutil.rmtree(logs)
    elif logs.exists() or logs.is_symlink():
        logs.unlink()
    edit = git_dir / "COMMIT_EDITMSG"
    if edit.exists() or edit.is_symlink():
        edit.unlink()


def make_release(
    work: Path,
    *,
    version: str,
    author_name: str,
    author_email: str,
    when: str,
    sign: bool,
    blank_config: Path,
    hooks: Path,
) -> tuple[str, str]:
    env = release_env(author_name, author_email, when, blank_config)
    quiet_log = ["-c", "core.logAllRefUpdates=false", "-c", f"core.hooksPath={hooks}"]
    git_run(["-C", str(work), *quiet_log, "init", "-b", "main"], env=env)
    git_run(["-C", str(work), "config", "core.autocrlf", "false"], env=env)
    git_run(["-C", str(work), "config", "core.logAllRefUpdates", "false"], env=env)
    common = ["-C", str(work), *quiet_log, *identity_args(author_name, author_email)]
    git_run([*common, "add", "-A"], env=env)
    message = release_message(version)
    git_run([*common, "commit", "-m", message], env=env)
    body = git_text(["-C", str(work), *quiet_log, "log", "-1", "--format=%B"], env=env)
    if body.strip() != message or "co-authored-by:" in body.lower():
        git_run([*common, "commit", "--amend", "-m", strip_coauthored(body) or message], env=env)
        body = git_text(["-C", str(work), *quiet_log, "log", "-1", "--format=%B"], env=env)
    if body.strip() != message or "co-authored-by:" in body.lower():
        raise ExportError("release commit message did not keep only the release text")
    ident = git_text(["-C", str(work), *quiet_log, "log", "-1", "--format=%an%n%ae%n%cn%n%ce"], env=env).splitlines()
    if ident != [author_name, author_email, author_name, author_email]:
        raise ExportError("release commit identity does not match the supplied author")
    tag = ["tag", "-s" if sign else "-a", f"v{version}", "-m", tag_message(version)]
    tag_sign = "tag.gpgSign=true" if sign else "tag.gpgSign=false"
    git_run(
        ["-C", str(work), *quiet_log, "-c", tag_sign, *identity_args(author_name, author_email), *tag],
        env=env,
    )
    tag_text = git_text(["-C", str(work), *quiet_log, "cat-file", "-p", f"v{version}"], env=env)
    tagger = next((line for line in tag_text.splitlines() if line.startswith("tagger ")), "")
    if "co-authored-by:" in tag_text.lower():
        raise ExportError("tag message included a Co-authored-by trailer")
    if not tagger.startswith(f"tagger {author_name} <{author_email}>") or not tagger.endswith("+0000"):
        raise ExportError("tag identity or date does not match the supplied author")
    if tag_message(version) not in tag_text:
        raise ExportError("tag message does not match the release text")
    commit_sha = git_text(["-C", str(work), *quiet_log, "rev-parse", "HEAD"], env=env).strip()
    tree = git_text(["-C", str(work), *quiet_log, "rev-parse", "HEAD^{tree}"], env=env).strip()
    scrub_git_dir(work)
    return commit_sha, tree


def public_layer(root: Path) -> list[str]:
    """Move the public-only files kept under ``public/`` to the same paths at the root."""
    layer = root / PUBLIC_LAYER
    if not layer.is_dir() or layer.is_symlink():
        return []
    moved: list[str] = []
    for path in sorted(layer.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ExportError(f"symlink in the public layer: {path.relative_to(root).as_posix()}")
        if not path.is_file():
            continue
        rel = path.relative_to(layer).as_posix()
        dest = root / rel
        if dest.exists() or dest.is_symlink():
            raise ExportError(f"public layer file already exists in the source tree: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.replace(dest)
        moved.append(rel)
    shutil.rmtree(layer)
    return sorted(moved)


_FLOW_BRANCHES = re.compile(r"^(\s*)branches:\s*\[.*\]\s*$", re.MULTILINE)


def _branch_entries(text: str) -> list[str]:
    """Entries of every ``branches:`` filter in a workflow file, flow or block style."""
    found: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)branches:\s*(.*?)\s*$", line)
        if not match:
            continue
        rest = match.group(2)
        if rest.startswith("["):
            found.extend(item.strip().strip("'\"") for item in rest.strip("[]").split(","))
            continue
        if rest and not rest.startswith("#"):
            found.append(rest.strip("'\""))
            continue
        indent = len(match.group(1))
        for follow in lines[index + 1:]:
            if not follow.strip():
                continue
            if len(follow) - len(follow.lstrip()) <= indent:
                break
            item = re.match(r"^\s*-\s*(.+?)\s*$", follow)
            if item:
                found.append(item.group(1).strip("'\""))
    return [entry for entry in found if entry]


def public_workflow_branches(root: Path) -> CheckResult:
    """Set each ci.yml branch filter to main, then require main-only filters in every workflow."""
    workflows = root / WORKFLOW_DIR
    ci = workflows / "ci.yml"
    if ci.is_file():
        text = ci.read_text(encoding="utf-8")
        new = _FLOW_BRANCHES.sub(lambda m: f"{m.group(1)}branches: [{PUBLIC_BRANCH}]", text)
        if new != text:
            ci.write_text(new, encoding="utf-8")
    bad: list[str] = []
    files = sorted(list(workflows.glob("*.yml")) + list(workflows.glob("*.yaml"))) if workflows.is_dir() else []
    for path in files:
        rel = path.relative_to(root).as_posix()
        for entry in _branch_entries(path.read_text(encoding="utf-8")):
            if entry != PUBLIC_BRANCH:
                bad.append(f"{rel}: {entry}")
    if bad:
        return CheckResult("workflow branches", "fail", "\n".join(bad))
    return CheckResult("workflow branches", "pass", f"every workflow branch filter is {PUBLIC_BRANCH} only")


def make_incremental(
    work: Path,
    *,
    parent_repo: Path,
    parent_ref: str,
    expect_parent: str,
    message: str,
    author_name: str,
    author_email: str,
    when: str,
    blank_config: Path,
    hooks: Path,
) -> tuple[str, str, str]:
    """One commit on top of ``parent_ref`` of a local clone. Returns (commit, tree, parent)."""
    env = release_env(author_name, author_email, when, blank_config)
    quiet = ["-c", "core.logAllRefUpdates=false", "-c", f"core.hooksPath={hooks}"]
    base = ["-C", str(work), *quiet]
    git_run([*base, "init", "-b", "main"], env=env)
    git_run(["-C", str(work), "config", "core.autocrlf", "false"], env=env)
    git_run(["-C", str(work), "config", "core.logAllRefUpdates", "false"], env=env)
    # Local directory only (validated in prepare_paths); no network remote.
    git_run([*base, "-c", "protocol.allow=never", "-c", "protocol.file.allow=always",
             "fetch", "--no-tags", "--quiet", str(parent_repo), parent_ref], env=env)
    parent = git_text([*base, "rev-parse", "--verify", "FETCH_HEAD^{commit}"], env=env).strip()
    if expect_parent and not parent.startswith(expect_parent.lower()):
        raise ExportError(f"parent {parent} is not the expected {expect_parent}")
    common = [*base, *identity_args(author_name, author_email)]
    git_run([*common, "add", "-A"], env=env)
    tree = git_text([*base, "write-tree"], env=env).strip()
    parent_tree = git_text([*base, "rev-parse", f"{parent}^{{tree}}"], env=env).strip()
    if tree == parent_tree:
        raise ExportError("export tree matches the parent; nothing to commit")
    commit = git_text([*common, "commit-tree", tree, "-p", parent, "-m", message], env=env).strip()
    git_run([*base, "update-ref", "refs/heads/main", commit], env=env)
    body = git_text([*base, "log", "-1", "--format=%B", commit], env=env)
    if body.strip() != message or "co-authored-by:" in body.lower():
        raise ExportError("sync commit message did not keep only the given text")
    ident = git_text([*base, "log", "-1", "--format=%an%n%ae%n%cn%n%ce%n%P", commit], env=env).splitlines()
    if ident != [author_name, author_email, author_name, author_email, parent]:
        raise ExportError("sync commit identity or parent does not match")
    scrub_git_dir(work)
    return commit, tree, parent


def remove_git(root: Path) -> None:
    git_dir = root / ".git"
    if git_dir.is_dir():
        shutil.rmtree(git_dir)
    elif git_dir.exists() or git_dir.is_symlink():
        git_dir.unlink()


def render_report(outcome: Outcome) -> str:
    excluded = "\n".join(f"- {path}" for path in outcome.excluded) or "- (none)"
    check_blocks = []
    for item in outcome.checks:
        body = item.detail.strip() or "(no output)"
        check_blocks.append(f"### {item.name}\n\n{item.status}\n\n{body}")
    checks = "\n\n".join(check_blocks)
    commit = outcome.commit_sha or "(none)"
    tree = outcome.git_tree or "(none)"
    command = push_command(outcome.public_url, outcome.branch)
    parent = outcome.parent_sha or "(none; root commit)"
    layer = "\n".join(f"- {path}" for path in outcome.public_layer) or "- (none)"
    return f"""# Export report

Source ref: {outcome.source_ref}
Parent commit: {parent}
Version: {outcome.version}
File count: {outcome.file_count}
Tree sha256: {outcome.tree_sha256}
Git commit: {commit}
Git tree: {tree}
Author: {outcome.author_name} <{outcome.author_email}>
Committer: {outcome.author_name} <{outcome.author_email}>
Commit date: {outcome.commit_date}
Result: {"pass" if outcome.ok else "fail"}

This tool did not push and did not create a remote repository.
This report is written beside the export directory and is not part of the release commit.
Identity denylist: {"used" if outcome.denylist_used else "not used"}
Tree sha256 is the SHA-256 of sorted lines `<file sha256><two spaces><relative path>`, each ending in a newline. `.git` is omitted.

## Excluded paths

{excluded}

## Public layer

{layer}

## Checks

{checks}

## Push command

Run this from the export directory when a person is ready to publish. This tool does not run it.

{command}
"""


def require_text(value: str, label: str) -> str:
    text = value.strip()
    if not text or any(ord(ch) < 32 for ch in text):
        raise ExportError(f"{label} must be a single line")
    return text


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public Sparksi export commit. Does not push.")
    parser.add_argument("--ref", required=True, help="Source commit to archive")
    parser.add_argument("--out", required=True, help="New directory for the export")
    parser.add_argument("--author-name", required=True, help="Author and committer name for the release commit")
    parser.add_argument("--author-email", required=True, help="Author and committer email for the release commit")
    parser.add_argument("--version", required=True, help="Release version; must match pyproject.toml in the ref")
    parser.add_argument("--repo", default="", help="Source git repository. Defaults to the current directory")
    parser.add_argument("--public-url", default="<public-url>", help="Remote URL printed in the push command")
    parser.add_argument("--exclude", default="", help="Exclude list. Defaults to scripts/export_exclude.txt")
    parser.add_argument("--checks-dir", default="", help="Directory of the guard scripts. Defaults to this script's directory")
    parser.add_argument("--add-spdx", action="store_true", help="Insert Apache-2.0 SPDX headers into the export only")
    parser.add_argument("--sign", action="store_true", help="Sign the annotated tag. The default tag is unsigned")
    parser.add_argument(
        "--allow-no-metadata-check",
        action="store_true",
        help="Continue when check_identifiers.py has no --git-metadata flag",
    )
    parser.add_argument("--parent-repo", default="", help="Local clone of the public repo. Makes one commit on --parent-ref; no tag")
    parser.add_argument("--parent-ref", default="main", help="Ref in --parent-repo to build on. Default main")
    parser.add_argument("--expect-parent", default="", help="Fail unless --parent-ref resolves to this sha (prefix allowed)")
    parser.add_argument("--branch", default="", help="Branch named in the printed push command. Default sync/<version>")
    parser.add_argument("--message", default="", help="Commit message for an incremental export")
    parser.add_argument(
        "--require-gitleaks",
        action="store_true",
        help="Fail when gitleaks is not on PATH. Set this on the final run",
    )
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    args.author_name = require_text(args.author_name, "author name")
    args.author_email = require_text(args.author_email, "author email")
    if "@" not in args.author_email or any(ch.isspace() for ch in args.author_email):
        raise ExportError("author email must be one address")
    if args.author_email.strip().lower() not in allowed_author_emails():
        raise ExportError("author email is not on SPARKSI_ALLOWED_AUTHORS")
    args.version = require_text(args.version, "version")
    if not VERSION_RE.fullmatch(args.version):
        raise ExportError("version must look like 0.7.0")
    args.public_url = require_text(args.public_url, "public url")
    if any(ch.isspace() for ch in args.public_url):
        raise ExportError("public url must be one token")
    if args.parent_repo:
        args.parent_ref = require_text(args.parent_ref, "parent ref")
        if args.parent_ref.startswith("-"):
            raise ExportError("parent ref must not start with -")
        args.branch = require_text(args.branch or f"sync/{args.version}", "branch")
        if args.branch.startswith("-") or any(ch.isspace() for ch in args.branch):
            raise ExportError("branch must be one token")
        args.message = require_text(args.message or sync_message(args.version), "message")
        if args.expect_parent and not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.expect_parent):
            raise ExportError("expect-parent must be a commit sha")
    elif args.expect_parent or args.branch or args.message:
        raise ExportError("--expect-parent, --branch and --message need --parent-repo")


def report_path(out: Path) -> Path:
    """Sibling of the export directory, so the report is not part of the release."""
    return out.parent / REPORT_NAME


def _outside(path: Path, repo: Path) -> bool:
    try:
        path.resolve().relative_to(repo.resolve())
    except ValueError:
        return True
    return False


def prepare_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve() if args.repo else Path.cwd().resolve()
    out = Path(args.out).resolve()
    report = report_path(out)
    checks = Path(args.checks_dir).resolve() if args.checks_dir else script_dir
    exclude = Path(args.exclude).resolve() if args.exclude else script_dir / "export_exclude.txt"
    if not (repo / ".git").exists():
        raise ExportError(f"source is not a git repository: {repo}")
    if args.parent_repo:
        parent = Path(args.parent_repo).resolve()
        if not parent.is_dir() or not ((parent / ".git").exists() or (parent / "HEAD").is_file()):
            raise ExportError(f"parent repo is not a local git directory: {parent}")
        args.parent_repo = str(parent)
    if out.exists():
        raise ExportError(f"output path already exists: {out}")
    if report.exists():
        raise ExportError(f"report path already exists: {report}")
    if not _outside(out, repo):
        raise ExportError("output path must be outside the source repository")
    if not _outside(report, repo):
        raise ExportError("report path must be outside the source repository")
    for name in CHECKERS:
        if not (checks / name).is_file():
            raise ExportError(f"missing checker: {checks / name}")
    return repo, out, checks, exclude


def metadata_availability(script: Path, allow_missing: bool) -> CheckResult | None:
    """None means the checker has ``--git-metadata`` and must run after the commit."""
    if checker_supports_git_metadata(script):
        return None
    if allow_missing:
        return CheckResult(
            "identifiers --git-metadata",
            "warn",
            "checker has no --git-metadata flag; metadata scan skipped because --allow-no-metadata-check was set",
        )
    return CheckResult("identifiers --git-metadata", "fail", "checker has no --git-metadata flag")


def build(work: Path, args: argparse.Namespace, repo: Path, checks: Path, exclude: Path, scratch: Path) -> Outcome:
    rev = resolve_ref(repo, args.ref)
    when = to_utc_git_date(ref_commit_date(repo, rev))
    archive_tree(repo, rev, work)
    excluded = apply_excludes(work, load_patterns(exclude))
    layer = public_layer(work)
    branches = public_workflow_branches(work)
    inserted = add_spdx_headers(work) if args.add_spdx else []
    meta_script = checks / "check_identifiers.py"
    early_meta = metadata_availability(meta_script, args.allow_no_metadata_check)
    results = [
        spdx_check(work, inserted),
        verify_logos(repo, rev, work),
        version_check(work, args.version),
        check_release_identity(args.author_name, args.author_email, args.version, args.message),
        branches,
        run_check("identifiers", [sys.executable, str(meta_script), str(work)], work),
        run_check("ui colors", [sys.executable, str(checks / "check_ui_colors.py"), str(work)], work),
        run_check("stdlib", [sys.executable, str(checks / "check_stdlib.py"), str(work)], work),
        run_gitleaks(work, required=args.require_gitleaks),
    ]
    if early_meta is not None:
        results.append(early_meta)
    commit_sha = ""
    git_tree = ""
    parent_sha = ""
    if any(item.failed for item in results):
        if early_meta is None:
            results.append(CheckResult(
                "identifiers --git-metadata",
                "skipped",
                "not run; an earlier check failed, so no commit was created",
            ))
    else:
        blank = scratch / "gitconfig"
        blank.write_text("", encoding="utf-8")
        hooks = scratch / "hooks"
        hooks.mkdir()
        try:
            if args.parent_repo:
                commit_sha, git_tree, parent_sha = make_incremental(
                    work,
                    parent_repo=Path(args.parent_repo),
                    parent_ref=args.parent_ref,
                    expect_parent=args.expect_parent,
                    message=args.message,
                    author_name=args.author_name,
                    author_email=args.author_email,
                    when=when,
                    blank_config=blank,
                    hooks=hooks,
                )
            else:
                commit_sha, git_tree = make_release(
                    work,
                    version=args.version,
                    author_name=args.author_name,
                    author_email=args.author_email,
                    when=when,
                    sign=args.sign,
                    blank_config=blank,
                    hooks=hooks,
                )
        except ExportError as exc:
            remove_git(work)
            results.append(CheckResult("git commit", "fail", str(exc)))
            if early_meta is None:
                results.append(CheckResult(
                    "identifiers --git-metadata",
                    "skipped",
                    "not run; the commit was discarded",
                ))
        else:
            if early_meta is None:
                meta = run_check(
                    "identifiers --git-metadata",
                    [sys.executable, str(meta_script), str(work), "--git-metadata"],
                    work,
                )
                results.append(meta)
                if meta.failed:
                    remove_git(work)
                    commit_sha = ""
                    git_tree = ""
            if (work / ".git").is_dir():
                scrub_git_dir(work)
    return Outcome(
        source_ref=rev,
        version=args.version,
        author_name=args.author_name,
        author_email=args.author_email,
        commit_date=when,
        file_count=len(export_files(work)),
        tree_sha256=tree_sha256(work),
        excluded=excluded,
        checks=results,
        commit_sha=commit_sha,
        git_tree=git_tree,
        public_url=args.public_url,
        denylist_used=bool(identity_denylist()),
        parent_sha=parent_sha,
        branch=args.branch if args.parent_repo else "",
        message=args.message,
        public_layer=layer,
    )


def publish(work: Path, out: Path, outcome: Outcome) -> None:
    report = render_report(outcome)
    destination = report_path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if outcome.ok:
        scrub_git_dir(work)
        shutil.move(str(work), str(out))
    destination.write_text(report, encoding="utf-8")


def export(args: argparse.Namespace) -> int:
    # Checkers this tool runs may call git. They must not lazy-fetch either.
    os.environ["GIT_NO_LAZY_FETCH"] = "1"
    validate(args)
    repo, out, checks, exclude = prepare_paths(args)
    scratch = Path(tempfile.mkdtemp(prefix="sparksi-export-meta-"))
    work = Path(tempfile.mkdtemp(prefix="sparksi-export-"))
    published = False
    try:
        outcome = build(work, args, repo, checks, exclude, scratch)
        publish(work, out, outcome)
        published = outcome.ok
        command = push_command(args.public_url, args.branch if args.parent_repo else "")
        print(f"report: {report_path(out)}")
        print(f"result: {'pass' if outcome.ok else 'fail'}")
        print(f"push command (not run): {command}")
        return 0 if outcome.ok else 1
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        if not published and out.exists():
            shutil.rmtree(out, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    try:
        return export(parse_args(argv))
    except ExportError as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
