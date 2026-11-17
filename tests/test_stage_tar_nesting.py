# SPDX-License-Identifier: Apache-2.0

"""A release tar packed as <name>/... unpacks into dest, not dest/<name>."""

import hashlib
import io
import tarfile

from sparksi import agent


def _tar(tmp_path, entries, links=()):
    path = tmp_path / "release.tar"
    with tarfile.open(path, "w") as tf:
        for name, data in entries:
            if data is None:
                info = tarfile.TarInfo(name)
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tf.addfile(info)
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE
            info.linkname = target
            tf.addfile(info)
    return path


def test_top_folder_named_like_dest_is_dropped(tmp_path):
    dest = tmp_path / "weights" / "bge-m3"
    tar = _tar(tmp_path, [("bge-m3", None), ("bge-m3/config.json", b"{}"), ("bge-m3/onnx/model.onnx", b"w")])
    agent._extract_release(tar, dest)
    assert (dest / "config.json").read_bytes() == b"{}"
    assert (dest / "onnx" / "model.onnx").read_bytes() == b"w"
    assert not (dest / "bge-m3").exists()


def test_files_at_the_root_unpack_unchanged(tmp_path):
    dest = tmp_path / "weights" / "bge-m3"
    tar = _tar(tmp_path, [("config.json", b"{}"), ("onnx/model.onnx", b"w")])
    agent._extract_release(tar, dest)
    assert (dest / "config.json").is_file()
    assert (dest / "onnx" / "model.onnx").is_file()


def test_a_top_folder_with_another_name_is_kept(tmp_path):
    dest = tmp_path / "weights" / "bge-m3"
    tar = _tar(tmp_path, [("other/config.json", b"{}")])
    agent._extract_release(tar, dest)
    assert (dest / "other" / "config.json").is_file()


def test_mixed_root_keeps_the_folder(tmp_path):
    dest = tmp_path / "weights" / "bge-m3"
    tar = _tar(tmp_path, [("bge-m3/config.json", b"{}"), ("README.md", b"r")])
    agent._extract_release(tar, dest)
    assert (dest / "bge-m3" / "config.json").is_file()
    assert (dest / "README.md").is_file()


def test_hard_link_under_the_top_folder_follows_it(tmp_path):
    dest = tmp_path / "weights" / "bge-m3"
    tar = _tar(tmp_path, [("bge-m3/a.bin", b"x")], links=[("bge-m3/b.bin", "bge-m3/a.bin")])
    agent._extract_release(tar, dest)
    assert (dest / "b.bin").read_bytes() == b"x"
    assert not (dest / "bge-m3").exists()


def test_stage_tar_lands_flat(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    archive.mkdir()
    tar = _tar(archive, [("bge-m3/config.json", b"{}")])
    digest = hashlib.sha256(tar.read_bytes()).hexdigest()
    dest = tmp_path / "weights" / "bge-m3"
    monkeypatch.setattr(agent, "archive_root", lambda: str(archive))
    monkeypatch.setattr(agent, "_weight_path", lambda raw: dest)
    monkeypatch.setattr(agent, "_free_ok", lambda *a, **k: None)
    got = agent._stage_tar({"tar": str(tar), "dest": str(dest), "sha256": digest})
    assert got["ok"] is True, got
    assert (dest / "config.json").is_file()
    assert not (dest / "bge-m3").exists()
