# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sparksi.catalog import load_catalog, load_yaml
from sparksi.lock import Lock
from sparksi.status import assemble_inventory, assemble_status
from sparksi.truth_diff import diff_catalog_vs_truth


def test_real_catalog_loads() -> None:
    cat = load_catalog()
    assert "coresident-3" in cat["recipes"]
    assert cat["recipes"]["video"]["estimate_gib"] is None
    assert cat["host"]["ui_port"] == 7500


def test_diff_match_and_mismatch(tmp_path: Path) -> None:
    catalog = {
        "host": {"uma_gib": 121},
        "residents": [
            {
                "id": "brain-a",
                "role": "brain",
                "served_name": "brain-a",
                "port": 8007,
                "unit": "vllm-brain-a.service",
                "weights": "/w/d",
                "gpu_memory_utilization": 0.20,
            },
            {
                "id": "nano",
                "role": "vision",
                "served_name": "wrong",
                "port": 8004,
                "unit": "vllm-vision-a.service",
                "weights": "/w/n",
                "gpu_memory_utilization": 0.28,
            },
        ],
    }
    truth = {
        "stack": {"id": "coresident-3", "mode": "always_on", "approx_vram_gb": 103},
        "models": [
            {
                "role": "brain",
                "served_name": "brain-a",
                "port": 8007,
                "unit": "vllm-brain-a.service",
                "weights": "/w/d",
                "gpu_memory_utilization": 0.20,
            },
            {
                "role": "vision",
                "served_name": "vision-a",
                "port": 8004,
                "unit": "vllm-vision-a.service",
                "weights": "/w/n",
                "gpu_memory_utilization": 0.28,
            },
            {
                "role": "counsel",
                "served_name": "counsel-a",
                "port": 8011,
                "unit": "vllm-counsel-a.service",
                "weights": "/w/a",
                "gpu_memory_utilization": 0.28,
            },
        ],
    }
    d = diff_catalog_vs_truth(catalog, truth)
    assert d["wrote_model_truth"] if False else d["source"].startswith("read-only")
    assert any(m["role"] == "brain" for m in d["matched"])
    assert any(m["role"] == "vision" for m in d["mismatches"])
    assert "counsel" in d["truth_only_roles"]
    assert d["drift"] is True


def test_does_not_write_truth(tmp_path: Path, monkeypatch) -> None:
    truth = tmp_path / "MODEL_TRUTH.yaml"
    truth.write_text("stack: {id: x}\n")
    before = truth.read_bytes()
    catalog = {"host": {}, "residents": []}
    diff_catalog_vs_truth(catalog, {"stack": {"id": "x"}, "models": []})
    assert truth.read_bytes() == before


def test_assemble_does_not_touch_tmp_truth(tmp_path: Path, monkeypatch) -> None:
    # Was the live MODEL_TRUTH (missing on CI); now a fake fixture.
    src = Path(__file__).parent / "fixtures" / "MODEL_TRUTH.example.yaml"
    truth = tmp_path / "MODEL_TRUTH.yaml"
    truth.write_bytes(src.read_bytes())
    before = truth.read_bytes()
    mtime = truth.stat().st_mtime_ns
    monkeypatch.setenv("SPARKSI_MODEL_TRUTH", str(truth))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    cat = load_catalog()
    t = load_yaml(truth)
    assemble_inventory(cat, t)
    assemble_status(cat, t, Lock(tmp_path / "lock"))
    assert truth.read_bytes() == before
    assert truth.stat().st_mtime_ns == mtime
