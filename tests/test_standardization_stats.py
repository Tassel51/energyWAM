"""Paper §4.8 standardized statistics: hand-computed checks.

Every test here asserts *numbers*, not just that a function exists:

* hand-computed population mean/std (masked and unmasked),
* mask/NaN exclusion, per-channel counts,
* zero-variance dimensions finite + invertible,
* float64 accumulator stability,
* exact round trip including the explicit ``epsilon``,
* dimension / provenance / fingerprint rejection,
* train-only manifest guards (val/test rejected, cross-split overlap rejected),
* the N(0, I) prior in the standardized space (fixed seed, tolerance),
* config validation of the approved baseline.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from .harness import TestFailure, assert_close, assert_true


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _stats(arr, mask=None, dim=None, epsilon=1e-6):
    from energywam.data.standardization import BundleFitter

    a = np.asarray(arr, dtype=np.float64)
    f = BundleFitter(action_dim=dim or a.shape[-1], future_dim=None, epsilon=epsilon)
    f.update_action(a, mask=mask)
    return f.action.finalize(epsilon)


def _bundle(eps=1e-6, **prov):
    from energywam.data.standardization import BundleFitter

    f = BundleFitter(action_dim=2, future_dim=3, epsilon=eps, provenance=dict(prov))
    f.update_action(np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]))
    f.update_future(np.asarray([[[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]]))
    return f.finalize(coverage={"kind": "synthetic_train_split"})


# ---------------------------------------------------------------------------
# 1. hand-computed statistics
# ---------------------------------------------------------------------------
def test_hand_computed_population_stats():
    """mean=[3, 30], population std=sqrt(8/3) and sqrt(800/3), counts=3."""
    x = np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    s = _stats(x)
    assert_close(s.mean, [3.0, 30.0], 1e-12, "mean")
    want_std = math.sqrt(8.0 / 3.0)
    assert_close(s.std[0], want_std, 1e-12, "population std dim0")
    assert_close(s.std[1], math.sqrt(800.0 / 3.0), 1e-12, "population std dim1")
    assert_close(s.count, [3.0, 3.0], 0, "counts")

    # hand-computed standardization of x[2, 0] = 5
    z = s.normalize(x)
    want = (5.0 - 3.0) / (want_std + 1e-6)
    assert_close(z[2, 0], want, 1e-12, "z = (x-mean)/(std+eps) hand value")
    # exact inverse
    back = s.denormalize(z)
    assert_close(back, x, 1e-9, "round trip")


def test_masked_elements_excluded_and_partial_channel_counts():
    """A masked element contributes to nothing; per-channel counts are exact."""
    x = np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    mask = np.asarray([[True, True], [False, True], [True, True]])
    s = _stats(x, mask=mask)
    # dim0 sees only 1 and 5 -> mean 3, population var 4 -> std 2
    assert_close(s.mean[0], 3.0, 1e-12, "masked mean dim0")
    assert_close(s.std[0], 2.0, 1e-12, "masked std dim0")
    assert_close(s.count, [2.0, 3.0], 0, "per-channel valid counts")
    assert_close(s.mean[1], 30.0, 1e-12, "unmasked dim1 mean")


def test_nan_elements_excluded():
    x = np.asarray([[1.0, 10.0], [float("nan"), 30.0], [5.0, 50.0]])
    s = _stats(x)
    assert_close(s.mean[0], 3.0, 1e-12, "NaN must not enter the statistics")
    assert_close(s.count, [2.0, 3.0], 0, "NaN excluded from the count")


def test_reduced_future_mask_pools_over_fvp():
    """[B, F, V, P] masks broadcast over the channel dim (F,V,P pooling)."""
    from energywam.data.standardization import BundleFitter

    z = np.arange(2 * 1 * 2 * 2 * 4, dtype=np.float64).reshape(2, 1, 2, 2, 4)
    mask = np.ones((2, 1, 2, 2), dtype=bool)
    mask[1, 0, 1, 1] = False  # one (B,F,V,P) position invalid
    f = BundleFitter(action_dim=2, future_dim=4, epsilon=1e-6)
    f.update_future(z, mask=mask)
    s = f.future.finalize()
    # every channel saw 2*1*2*2 - 1 = 7 valid elements
    assert_close(s.count, [7.0] * 4, 0, "per-channel counts after F,V,P pooling")
    # hand mean over the valid positions for channel 0
    assert_close(s.mean[0], z[..., 0][mask].mean(), 1e-12, "pooled channel mean")


# ---------------------------------------------------------------------------
# 2. zero variance / float64 stability
# ---------------------------------------------------------------------------
def test_zero_variance_finite_and_invertible():
    x = np.asarray([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    s = _stats(x)
    assert_close(s.std[0], 0.0, 0, "constant column has population std 0")
    z = s.normalize(x)
    assert_true(np.isfinite(z).all(), "zero-variance normalization must stay finite")
    assert_close(z[:, 0], [0.0, 0.0, 0.0], 1e-12, "constant column maps to 0 (its mean)")
    back = s.denormalize(z)
    assert_close(back, x, 1e-9, "zero-variance round trip must return the input")
    # the constant column is NOT treated as raw: x=3 would map to 1e6 * eps-scale
    z3 = s.normalize(np.asarray([[3.0, 0.0]]))
    assert_close(z3[0, 0], (3.0 - 2.0) / (0.0 + 1e-6), 1e-6, "divisor is exactly (std+eps)")


def test_float64_streaming_stability():
    """Large offset + small spread: the float64 accumulator keeps the spread."""
    n = 20000
    x = 1.0e9 + np.arange(n, dtype=np.float64)
    s = _stats(x.reshape(-1, 1))
    # exact population variance of 0..n-1 is (n^2-1)/12
    want = math.sqrt((n * n - 1) / 12.0)
    assert_close(s.std[0], want, 1e-6 * want, "float64 Welford variance")
    assert_close(s.mean[0], 1.0e9 + (n - 1) / 2.0, 1e-3, "float64 mean")


# ---------------------------------------------------------------------------
# 3. coordinate definition / round trip / adapter
# ---------------------------------------------------------------------------
def test_round_trip_and_explicit_epsilon():
    rng = np.random.RandomState(0)
    x = rng.randn(64, 5) * 3.0 - 1.0
    s = _stats(x, epsilon=1e-6)
    z = s.normalize(x)
    # formula is exactly (x - mean) / (std + epsilon)
    want = (x - s.mean) / (s.std + 1e-6)
    assert_close(z, want, 1e-12, "coordinate formula")
    assert_close(s.denormalize(z), x, 1e-9, "round trip (float64)")

    x32 = x.astype("float32")
    s32 = _stats(x32, epsilon=1e-6)
    z32 = s32.normalize(x32)
    back32 = s32.denormalize(z32)
    assert_close(back32, x32, 1e-4, "round trip (float32 tolerance)")

    # epsilon is part of the definition: a different epsilon changes the map
    s2 = _stats(x, epsilon=1e-3)
    assert_true(float(np.abs(s2.normalize(x) - z).max()) > 0, "epsilon must matter")


def test_normstats_zscore_adapter_matches():
    from energywam.data.normalization import normalize, unnormalize

    x = np.linspace(-2, 2, 24).reshape(8, 3).astype("float32")
    s = _stats(x.astype("float64"), epsilon=1e-6)
    ns = s.to_norm_stats()
    assert_true(ns.mode == "zscore", f"adapter mode {ns.mode}")
    assert_close(ns.epsilon, 1e-6, 0, "adapter carries epsilon")
    a = normalize(x, ns)
    b = s.normalize(x)
    assert_close(a, b, 1e-6, "NormStats zscore == ZScoreStats normalize")
    assert_close(unnormalize(a, ns), x, 1e-4, "NormStats zscore inverse")
    # zero-variance column keeps working through the adapter
    xs = np.ones((4, 3), dtype="float32")
    ss = _stats(xs.astype("float64"))
    ns2 = ss.to_norm_stats()
    zs = normalize(xs, ns2)
    assert_true(np.isfinite(zs).all(), "zscore adapter finite on zero variance")
    assert_close(unnormalize(zs, ns2), xs, 1e-5, "zero-variance adapter inverse")


# ---------------------------------------------------------------------------
# 4. rejection paths
# ---------------------------------------------------------------------------
def test_dimension_mismatch_rejected():
    s = _stats(np.zeros((4, 3)) + np.arange(3))
    for fn in (s.normalize, s.denormalize):
        try:
            fn(np.zeros((4, 2)))
            raise TestFailure(f"{fn.__name__} accepted a wrong last dim")
        except Exception as exc:
            assert_true("dim" in str(exc), f"unhelpful error: {exc}")


def test_provenance_mismatch_rejected():
    b = _bundle(
        camera_order=["cam0", "cam1"],
        action_horizon=8,
        future_offsets=[3, 5, 8],
        dino_preprocess_version="dino_v3_vits16_224_center_norm_v1",
        dino={"model_path": "/weights/dino"},
    )
    kw = dict(
        action_dim=2,
        future_dim=3,
        camera_order=["cam0", "cam1"],
        action_horizon=8,
        future_offsets=[3, 5, 8],
        dino_preprocess_version="dino_v3_vits16_224_center_norm_v1",
        dino_model_path="/weights/dino",
        epsilon=1e-6,
        require_future=True,
    )
    b.validate_geometry(**kw)  # consistent: must pass
    bad = [
        {"action_dim": 3},
        {"future_dim": 4},
        {"camera_order": ["cam1", "cam0"]},
        {"action_horizon": 16},
        {"future_offsets": [2, 4, 6]},
        {"dino_preprocess_version": "other_v"},
        {"dino_model_path": "/other/dino"},
        {"epsilon": 1e-3},
    ]
    for override in bad:
        k = dict(kw)
        k.update(override)
        try:
            b.validate_geometry(**k)
            raise TestFailure(f"validate_geometry accepted mismatch {override}")
        except TestFailure:
            raise
        except Exception as exc:
            assert_true("mismatch" in str(exc), f"unhelpful error for {override}: {exc}")

    # a future-less bundle is refused when future statistics are required
    from energywam.data.standardization import BundleFitter

    f = BundleFitter(action_dim=2, future_dim=None, epsilon=1e-6)
    f.update_action(np.zeros((3, 2)) + 1.0)
    b2 = f.finalize(coverage={"kind": "subset"})
    try:
        b2.validate_geometry(require_future=True, action_dim=2)
        raise TestFailure("future-less bundle accepted where future stats are required")
    except TestFailure:
        raise
    except Exception as exc:
        assert_true("future" in str(exc).lower(), f"unhelpful error: {exc}")
    b2.validate_geometry(require_future=False, action_dim=2)  # action_only: fine


def test_fingerprint_detects_modification():
    from energywam.data.standardization import NormalizationBundle, StatsError

    b = _bundle()
    payload = b.to_payload()
    assert_true(len(payload["fingerprint"]) == 64, "sha256 fingerprint")
    # identical content -> identical fingerprint
    b2 = NormalizationBundle.from_payload(payload)
    assert_true(b2.fingerprint == b.fingerprint, "fingerprint unstable")
    # tampering with the numbers is caught by the embedded fingerprint
    tampered = dict(payload)
    tampered["action"] = dict(payload["action"])
    tampered["action"]["mean"] = [v + 1.0 for v in payload["action"]["mean"]]
    try:
        NormalizationBundle.from_payload(tampered)
        raise TestFailure("tampered bundle payload accepted")
    except TestFailure:
        raise
    except StatsError as exc:
        assert_true("fingerprint" in str(exc), f"unhelpful error: {exc}")
    # changing provenance also changes the fingerprint
    b3 = _bundle(dataset_identity="other/ds")
    assert_true(b3.fingerprint != b.fingerprint, "provenance must enter the fingerprint")


def test_save_refuses_overwrite_and_load_missing():
    from energywam.data.standardization import StatsError, load_bundle, save_bundle

    b = _bundle()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "stats.json"
        save_bundle(b, p)
        try:
            save_bundle(b, p)
            raise TestFailure("overwrite was not refused")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("overwrite" in str(exc), f"unhelpful error: {exc}")
        save_bundle(b, p, overwrite=True)
        b2 = load_bundle(p)
        assert_true(b2.fingerprint == b.fingerprint, "bundle round trip")
        try:
            load_bundle(Path(d) / "missing.json")
            raise TestFailure("missing bundle did not raise")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("not found" in str(exc), f"unhelpful error: {exc}")


# ---------------------------------------------------------------------------
# 5. train-only manifest guards
# ---------------------------------------------------------------------------
class _MiniReader:
    dataset_identity = "synthetic/robot"
    episodes = [{"episode_index": 0, "length": 10}, {"episode_index": 1, "length": 10}]


def _manifest(path, split, eps, identity="synthetic/robot"):
    import json

    path.write_text(json.dumps({
        "schema_version": "energywam_episode_manifest_v1",
        "split": split,
        "dataset_identity": identity,
        "episode_indices": eps,
    }))
    return path


def test_manifest_rejects_val_test_and_overlap():
    from energywam.data.standardization import StatsError
    from energywam.data.stats_fit import load_train_manifest

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for bad_split in ("val", "test", "VAL", "evaluation"):
            p = _manifest(d / f"{bad_split}.json", bad_split, [0])
            try:
                load_train_manifest(p, reader=_MiniReader())
                raise TestFailure(f"split={bad_split!r} was accepted")
            except TestFailure:
                raise
            except StatsError as exc:
                assert_true("TRAIN" in str(exc) or "train" in str(exc), str(exc))

        train = _manifest(d / "train.json", "train", [0, 1])
        man, eps = load_train_manifest(train, reader=_MiniReader())
        assert_true(eps == [0, 1], f"episodes {eps}")
        assert_true(man["manifest_sha256"], "manifest hash recorded")

        # unknown episode -> rejected
        bad = _manifest(d / "train_bad.json", "train", [0, 7])
        try:
            load_train_manifest(bad, reader=_MiniReader())
            raise TestFailure("unknown episode accepted")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("not in the dataset" in str(exc), str(exc))

        # dataset identity mismatch -> rejected
        wrong = _manifest(d / "train_wrong.json", "train", [0], identity="other/ds")
        try:
            load_train_manifest(wrong, reader=_MiniReader())
            raise TestFailure("dataset identity mismatch accepted")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("dataset" in str(exc).lower(), str(exc))

        # cross-split overlap -> rejected
        val = _manifest(d / "val.json", "val", [1])
        try:
            load_train_manifest(train, reader=_MiniReader(), other_manifest_paths=[val])
            raise TestFailure("cross-split overlap accepted")
        except TestFailure:
            raise
        except StatsError as exc:
            assert_true("overlap" in str(exc), str(exc))


# ---------------------------------------------------------------------------
# 6. prior N(0, I) + config validation
# ---------------------------------------------------------------------------
def test_gaussian_prior_mean_and_scale():
    """init_action / init_future under the approved prior: mean 0, std 1."""
    import torch

    from energywam.config import InitConfig
    from energywam.energy.init import init_action, init_future

    cfg = InitConfig()
    cfg.init_action_mean, cfg.init_action_std = 0.0, 1.0
    cfg.init_future_mean, cfg.init_future_std = 0.0, 1.0
    g = torch.Generator().manual_seed(1234)
    A = init_action(2000, 8, 3, cfg, device=torch.device("cpu"), generator=g)
    Z = init_future(2000, 3, 1, 4, 8, cfg, device=torch.device("cpu"), generator=g)
    for name, t in (("A0", A), ("Z0", Z)):
        v = t.double().numpy().reshape(-1)
        assert_close(float(v.mean()), 0.0, 0.02, f"{name} prior mean")
        assert_close(float(v.std()), 1.0, 0.02, f"{name} prior std")
    # distinct draws (not a constant / degenerate stream)
    assert_true(float(A.std()) > 0.9, "prior draws must actually be random")


def test_config_enforces_paper_prior_and_profile():
    from energywam.config import config_from_dict
    from energywam.data.standardization import StatsError

    base = {
        "backbone": {"backend": "tiny"},
        "data": {
            "normalization_profile": "paper_standardized_v1",
            "action_normalization": "zscore",
        },
        "init": {"init_action_std": 1.0, "init_future_std": 1.0},
    }
    config_from_dict(base)  # valid baseline
    for over, expect in (
        ({"init": {"init_action_std": 0.1}}, "N(0, I)"),
        ({"init": {"init_future_std": 0.5}}, "N(0, I)"),
        ({"data": {"action_normalization": "min_max"}}, "zscore"),
        ({"data": {"normalization_profile": "legacy"}, "init": {}}, None),
    ):
        import copy

        d = copy.deepcopy(base)
        for k, v in over.items():
            d[k].update(v)
        try:
            config_from_dict(d)
            if expect is None:
                raise TestFailure(f"config accepted invalid combo {over}")
            raise TestFailure(f"config accepted {over}")
        except TestFailure:
            raise
        except ValueError as exc:
            if expect is not None:
                assert_true(expect in str(exc), f"unhelpful error: {exc}")

    # explicit prior ablation switch allows a non-(0,1) prior
    d = copy.deepcopy(base)
    d["init"] = {"init_action_std": 0.5, "allow_prior_ablation": True}
    config_from_dict(d)

    # legacy profile must not claim zscore / a bundle path
    try:
        config_from_dict({"data": {"action_normalization": "zscore"}})
        raise TestFailure("legacy profile accepted action_normalization=zscore")
    except TestFailure:
        raise
    except ValueError as exc:
        assert_true("paper_standardized_v1" in str(exc), str(exc))
    try:
        config_from_dict({"data": {"norm_stats_bundle_path": "/tmp/x.json"}})
        raise TestFailure("legacy profile accepted norm_stats_bundle_path")
    except TestFailure:
        raise
    except ValueError as exc:
        assert_true("paper_standardized_v1" in str(exc), str(exc))


def test_current_dino_noise_maps_raw_base_into_standardized_space():
    """The optional init maps raw current DINO through the FUTURE statistics."""
    import torch

    from energywam.config import InitConfig
    from energywam.data.standardization import ZScoreStats
    from energywam.energy.init import init_future

    stats = ZScoreStats(mean=np.asarray([10.0, 0.0]), std=np.asarray([2.0, 4.0]),
                        count=np.asarray([5.0, 5.0]), epsilon=1e-6)
    cfg = InitConfig()
    cfg.initialization_future = "current_dino_noise"
    cfg.init_future_noise_std = 0.0  # isolate the base
    cd = torch.tensor([[[[10.0, 0.0], [12.0, 4.0]]]])  # [1, 1, P=2, D=2]
    z = init_future(1, 3, 1, 2, 2, cfg, device=torch.device("cpu"), current_dino=cd,
                    future_stats=stats)
    want = torch.tensor([0.0, 0.0])   # (10-10)/(2+eps), (0-0)/(4+eps)
    want2 = torch.tensor([1.0, 1.0])  # (12-10)/(2+eps), (4-0)/(4+eps)
    assert_close(z[0, 0, 0, 0].numpy(), want.numpy(), 1e-6, "standardized base 1")
    assert_close(z[0, 2, 0, 1].numpy(), want2.numpy(), 1e-6, "standardized base 2 (broadcast over F)")
    # noise is added in the standardized space
    cfg.init_future_noise_std = 0.5
    torch.manual_seed(0)
    z2 = init_future(4000, 3, 1, 2, 2, cfg, device=torch.device("cpu"),
                     current_dino=cd.expand(4000, 1, 2, 2), future_stats=stats)
    first_patch = z2[:, :, 0, 0, :].reshape(-1).double().numpy()  # standardized mean 0
    assert_close(float(first_patch.mean()), 0.0, 0.02, "noise base mean in standardized space")
    assert_close(float(first_patch.std()), 0.5, 0.02, "noise std in standardized space")


TESTS = [
    ("hand_computed_population_stats", test_hand_computed_population_stats),
    ("masked_elements_excluded_and_partial_channel_counts", test_masked_elements_excluded_and_partial_channel_counts),
    ("nan_elements_excluded", test_nan_elements_excluded),
    ("reduced_future_mask_pools_over_fvp", test_reduced_future_mask_pools_over_fvp),
    ("zero_variance_finite_and_invertible", test_zero_variance_finite_and_invertible),
    ("float64_streaming_stability", test_float64_streaming_stability),
    ("round_trip_and_explicit_epsilon", test_round_trip_and_explicit_epsilon),
    ("normstats_zscore_adapter_matches", test_normstats_zscore_adapter_matches),
    ("dimension_mismatch_rejected", test_dimension_mismatch_rejected),
    ("provenance_mismatch_rejected", test_provenance_mismatch_rejected),
    ("fingerprint_detects_modification", test_fingerprint_detects_modification),
    ("save_refuses_overwrite_and_load_missing", test_save_refuses_overwrite_and_load_missing),
    ("manifest_rejects_val_test_and_overlap", test_manifest_rejects_val_test_and_overlap),
    ("gaussian_prior_mean_and_scale", test_gaussian_prior_mean_and_scale),
    ("config_enforces_paper_prior_and_profile", test_config_enforces_paper_prior_and_profile),
    ("current_dino_noise_maps_raw_base_into_standardized_space", test_current_dino_noise_maps_raw_base_into_standardized_space),
]
