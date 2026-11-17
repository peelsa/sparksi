# SPDX-License-Identifier: Apache-2.0

from sparksi import agent


MOUNTS = (
    "systemd-1 /mnt/archive autofs rw 0 0\n"
    "//192.0.2.72/archive /mnt/archive cifs rw,addr=192.0.2.72,soft 0 0\n"
)


def test_cifs_addr_reads_mount_text() -> None:
    assert agent._cifs_addr(MOUNTS, "/mnt/archive") == "192.0.2.72"
    assert agent._cifs_addr(MOUNTS, "/mnt/other") is None


def test_probe_skips_dead_archive(monkeypatch) -> None:
    monkeypatch.setattr(agent, "_archive_reachable", lambda root: False)
    monkeypatch.setattr(agent, "_scan_releases", lambda: (_ for _ in ()).throw(OSError(112, "Host is down")))
    out = agent._probe(
        {
            "op": "probe",
            "ports": [],
            "paths": ["/tmp"],
            "scan_releases": True,
            "scan_flat": "/mnt/archive/models/glm",
            "shards": 2,
        }
    )
    assert out["ok"] is True
    assert out["mount_archive"] is False
    assert out["releases"] == []
    assert out["flat"]["error"] == "archive host down"
    assert out["paths"]["/tmp"]["exists"] is True


def test_mount_entry_skips_autofs_and_reads_nfs_and_bind() -> None:
    nfs = (
        "systemd-1 /mnt/archive autofs rw 0 0\n"
        "198.18.0.10:/mnt/archive-raw/archive /mnt/archive nfs4 rw,vers=4.2,addr=198.18.0.10 0 0\n"
    )
    src, fstype, opts = agent._mount_entry(nfs, "/mnt/archive")
    assert fstype == "nfs4" and agent._opt(opts, "addr") == "198.18.0.10"
    bind = "/dev/sda1 /mnt/archive ext4 rw,relatime 0 0\n"
    assert agent._mount_entry(bind, "/mnt/archive")[:2] == ("/dev/sda1", "ext4")
    assert agent._mount_entry("systemd-1 /mnt/archive autofs rw 0 0\n", "/mnt/archive") is None


def test_fstab_server_for_untriggered_automount() -> None:
    fstab = (
        "# //192.0.2.64/archive /mnt/archive cifs x 0 0\n"
        "198.18.0.10:/mnt/archive-raw/archive /mnt/archive nfs rw 0 0\n"
    )
    assert agent._fstab_server(fstab, "/mnt/archive") == ("198.18.0.10", 2049)
    assert agent._fstab_server("//192.0.2.65/archive /mnt/x cifs rw 0 0\n", "/mnt/x") == ("192.0.2.65", 445)


def test_das_paths_are_blocked_for_writes() -> None:
    import pytest

    with pytest.raises(ValueError):
        agent._weight_path("/mnt/archive-raw/archive/models/x")


def test_storage_rows_label_archive() -> None:
    from sparksi.library import storage

    probes = {
        "spark-a": {"ok": True, "disk_total_gib": 3755.0, "disk_used_gib": 1377.0, "disk_free_gib": 2187.0,
                       "disk_used_pct": 39.0, "mount_archive": True,
                       "archive_store": {"path": "/mnt/archive", "device": "/dev/sda1", "fstype": "ext4",
                                         "total_gib": 3666.0, "used_gib": 1485.0, "free_gib": 2181.0, "used_pct": 41.0}},
        "spark-b": {"ok": True, "disk_total_gib": 3755.0, "disk_free_gib": 2395.0, "mount_archive": True},
    }
    rows = storage({}, probes)
    arch = [r for r in rows if r["kind"] == "archive"][0]
    assert arch["label"] == "Archive (RAID 1 DAS)"
    assert arch["ok"] and arch["used_pct"] == 41.0
    assert arch["mounted_on"] == {"spark-a": True, "spark-b": True}
    assert not any("archive_tar" in r["label"].lower() for r in rows)
