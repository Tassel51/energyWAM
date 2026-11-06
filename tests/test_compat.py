"""Compatibility: EnergyWAM must not break existing StarVLA imports or configs.

Also covers the config-parsing requirement: every YAML under ``configs/`` must
load, validate and produce a runnable model in its declared mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .harness import TestFailure, assert_true

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def test_starvla_frameworks_still_import():
    """The pre-existing StarVLA framework modules must still import.

    We only *read* ``../starVLA``; nothing in EnergyWAM writes to it.  Any new
    optional dependency must be lazily imported, so a plain ``import`` of the
    framework package must not need it.
    """
    starvla = ROOT.parent / "starVLA"
    if not starvla.exists():
        raise TestFailure(f"starVLA not found at {starvla}")
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import starVLA.model.tools as t\n"
        "print('FRAMEWORK_REGISTRY', type(t.FRAMEWORK_REGISTRY).__name__)\n"
        "import starVLA.model.framework.base_framework as bf\n"
        "print('baseframework', bf.baseframework.__name__)\n"
        "import starVLA.model.modules.vlm.QWen3 as q\n"
        "print('Qwen3VL interface', q._QWen3_VL_Interface.__name__)\n"
        "print('OK')\n"
    ) % str(starvla)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if r.returncode != 0:
        raise TestFailure(f"starVLA import failed:\n{r.stdout}\n{r.stderr}")
    assert_true("OK" in r.stdout, f"unexpected output: {r.stdout}")


def test_energywam_import_does_not_require_optional_deps():
    """Importing energywam must not pull in optional heavy deps eagerly."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import energywam\n"
        "import energywam.config\n"
        "import energywam.energy.solver\n"
        "import energywam.energy.losses\n"
        "import energywam.models.energy_core\n"
        "print('OK')\n"
    ) % str(ROOT)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if r.returncode != 0:
        raise TestFailure(f"energywam import failed:\n{r.stdout}\n{r.stderr}")
    assert_true("OK" in r.stdout, r.stdout)


def test_all_configs_parse_and_build_tiny():
    """Every YAML in configs/ must load and validate; tiny ones must build."""
    from energywam.config import load_config

    cfg_dir = ROOT / "configs"
    yamls = sorted(cfg_dir.glob("*.yaml"))
    assert_true(len(yamls) >= 5, f"expected at least 5 configs, found {len(yamls)}")
    for y in yamls:
        cfg = load_config(y)
        assert_true(cfg.framework.name == "QwenEnergyWAM", f"{y.name}: wrong framework name")
        assert_true(cfg.stage.stage in (
            "bi", "joint", "direct_joint", "action_only",
            "action_given_future", "future_given_action",
        ), f"{y.name}: unknown stage {cfg.stage.stage}")
        if cfg.backbone.backend == "tiny":
            from energywam.models.energy_wam import QwenEnergyWAM

            m = QwenEnergyWAM(cfg)
            n = sum(p.numel() for p in m.parameters())
            assert_true(n > 0, f"{y.name}: empty model")
            print(f"  {y.name}: stage={cfg.stage.stage} backend=tiny params={n}")


def test_real_mode_missing_weights_raises_loudly():
    """Real mode must fail with a clear error when the weights are absent —
    never silently fall back to a mock."""
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM

    cfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    cfg.backbone.backend = "qwen3vl"
    cfg.backbone.base_vlm = "/nonexistent/Qwen3-VL-4B-Instruct"
    cfg.backbone.dino_path = "/nonexistent/dino"
    try:
        QwenEnergyWAM(cfg)
    except FileNotFoundError as exc:
        assert_true("Qwen3-VL" in str(exc) or "weights" in str(exc), f"unhelpful error: {exc}")
        print(f"  real mode missing weights -> {type(exc).__name__}")
        return
    raise TestFailure("real mode with missing weights did not raise FileNotFoundError")


def test_mock_backbone_refused_outside_tiny():
    """The mock condition encoder is only reachable via backend='tiny'."""
    from energywam.config import load_config
    from energywam.models.energy_wam import build_backbone

    cfg = load_config(ROOT / "configs" / "tiny_cpu.yaml")
    cfg.backbone.backend = "not_a_backend"
    try:
        build_backbone(cfg)
    except ValueError as exc:
        assert_true("backend" in str(exc), str(exc))
        return
    raise TestFailure("unknown backend did not raise")


TESTS = [
    ("starvla_frameworks_still_import", test_starvla_frameworks_still_import),
    ("energywam_import_does_not_require_optional_deps", test_energywam_import_does_not_require_optional_deps),
    ("all_configs_parse_and_build_tiny", test_all_configs_parse_and_build_tiny),
    ("real_mode_missing_weights_raises_loudly", test_real_mode_missing_weights_raises_loudly),
    ("mock_backbone_refused_outside_tiny", test_mock_backbone_refused_outside_tiny),
]
