"""Minimal test harness (pytest is not installed in the borrowed env).

Each ``tests/test_*.py`` module exposes a module-level ``TESTS`` list of
``(name, fn)`` pairs.  ``fn()`` takes no arguments and either returns normally
(pass) or raises (fail).  There is **no skip**: a test that cannot run must be
explicitly reported as not-executed in IMPLEMENTATION_STATUS.md, never counted
as a pass.

Usage::

    python tests/run_all.py                 # run every test_*.py
    python tests/run_all.py test_gradients  # run one module
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"


class TestFailure(AssertionError):
    pass


def assert_close(a, b, tol, msg="", equal_nan: bool = False):
    """Compare two scalars/arrays with an absolute tolerance.

    * float64 precision is preserved (no forced ``.float()`` downcast);
    * NaN on either side **fails** unless ``equal_nan=True`` — a NaN diff makes
      ``diff > tol`` false, which previously let broken comparisons pass.
    """
    import numpy as np
    import torch

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    aa = _to_np(a)
    bb = _to_np(b)
    if aa.shape != bb.shape:
        raise TestFailure(f"{msg} | shape mismatch {aa.shape} vs {bb.shape}")
    aa64 = aa.astype(np.float64, copy=False)
    bb64 = bb.astype(np.float64, copy=False)
    both_nan = np.isnan(aa64) & np.isnan(bb64)
    nan_either = np.isnan(aa64) | np.isnan(bb64)
    if nan_either.any():
        if not equal_nan:
            raise TestFailure(
                f"{msg} | NaN present in comparison "
                f"(a has {int(np.isnan(aa64).sum())}, b has {int(np.isnan(bb64).sum())}); "
                "pass equal_nan=True if NaN equality is intended"
            )
        if not bool(both_nan[nan_either].all()):
            raise TestFailure(f"{msg} | NaN mismatch (only one side is NaN)")
    if not np.isfinite(aa64[~both_nan]).all() or not np.isfinite(bb64[~both_nan]).all():
        raise TestFailure(f"{msg} | non-finite (Inf) value in comparison")
    diff = float(np.abs(aa64 - np.where(both_nan, 0.0, bb64)).max()) if aa64.size else 0.0
    if diff > tol:
        raise TestFailure(f"{msg} | max diff {diff:.3e} > tol {tol:.3e}")


def assert_close_finite(a, b, tol, msg=""):
    """Like :func:`assert_close` but explicitly requires finite values."""
    assert_close(a, b, tol, msg=msg, equal_nan=False)


def assert_true(cond, msg=""):
    if not cond:
        raise TestFailure(msg or "assertion failed")


def assert_nonzero_finite(t, msg=""):
    import torch

    if not torch.isfinite(t).all():
        raise TestFailure(f"{msg} | non-finite values present")
    if float(t.abs().sum()) == 0.0:
        raise TestFailure(f"{msg} | tensor is all zeros")


def collect_modules(filter_name: str | None = None):
    import pkgutil

    mods = []
    for _, name, _ in pkgutil.iter_modules([str(ROOT / "tests")]):
        if not name.startswith("test_"):
            continue
        if filter_name and filter_name not in name:
            continue
        mods.append(importlib.import_module(f"tests.{name}"))
    return mods


def run(filter_name: str | None = None) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    n_pass = n_fail = 0
    t0 = time.time()
    for mod in collect_modules(filter_name):
        tests = getattr(mod, "TESTS", None)
        if tests is None:
            print(f"[SKIP-MODULE] {mod.__name__} has no TESTS")
            continue
        for name, fn in tests:
            full = f"{mod.__name__}.{name}"
            s = time.time()
            try:
                fn()
                dt = time.time() - s
                n_pass += 1
                results.append({"test": full, "status": "pass", "seconds": round(dt, 3)})
                print(f"[PASS] {full} ({dt:.2f}s)")
            except Exception as exc:  # noqa: BLE001
                dt = time.time() - s
                n_fail += 1
                tb = traceback.format_exc()
                results.append(
                    {"test": full, "status": "fail", "seconds": round(dt, 3),
                     "error": repr(exc), "traceback": tb}
                )
                print(f"[FAIL] {full} ({dt:.2f}s): {exc}")
                print(tb)
    total = time.time() - t0
    summary = {
        "pass": n_pass,
        "fail": n_fail,
        "total": n_pass + n_fail,
        "seconds": round(total, 2),
        "results": results,
    }
    out = LOG_DIR / f"test_results{'_' + filter_name if filter_name else ''}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n== {n_pass} passed, {n_fail} failed, {total:.1f}s ==")
    print(f"-> {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(run(filt))


def _self_check_nan_must_fail():
    """Meta-test: assert_close must reject NaN.  Returns True if the guard works."""
    try:
        assert_close(float("nan"), 1.0, 1e-6, "nan-vs-num should fail")
    except TestFailure:
        pass
    else:
        return False
    try:
        assert_close(1.0, float("nan"), 1e-6, "num-vs-nan should fail")
    except TestFailure:
        pass
    else:
        return False
    try:
        assert_close(float("nan"), float("nan"), 1e-6, "nan-vs-nan without equal_nan")
    except TestFailure:
        pass
    else:
        return False
    # equal_nan=True allows matched NaNs
    assert_close(float("nan"), float("nan"), 1e-6, "matched nan", equal_nan=True)
    return True
