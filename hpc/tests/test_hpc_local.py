"""Local CPU validation for the HPC deployment artefacts.

Run::

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      /data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python hpc/tests/test_hpc_local.py

What is checked
---------------
1. both HPC YAMLs parse AND pass ``validate_config``;
2. the main smoke config keeps every scientific condition (language trained,
   LoRA off, full unroll, real geometry, float32 candidates, standardized
   coordinate, optimizer untouched);
3. the frozen-Qwen diagnostic differs from the main config in **exactly one
   key** (so it cannot silently become a different experiment);
4. ``hpc/scripts/make_episode_manifests.py`` produces an episode-disjoint,
   deterministic, loader-validated split on a real-format LeRobot tree.

NOTE ON DATA: the manifest generator is exercised against the repo's
``tests/synthetic_lerobot.py`` fixture (a real-format LeRobot tree with
deterministic synthetic frames).  That is a *test fixture*, not a dataset: the
generator itself has no synthetic mode and refuses any tree without
``meta/info.json``.  Nothing here is a real RoboCasa365 measurement.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "energywam").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

PY = sys.executable

PASS: list[str] = []
FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    (PASS if cond else FAIL).append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)


# ---------------------------------------------------------------------------
# 1 + 2 + 3: configuration contracts
# ---------------------------------------------------------------------------
def test_configs() -> None:
    print("[1-3] HPC configuration contracts")
    import dataclasses

    from energywam.config import load_config, to_plain_dict

    main_path = ROOT / "hpc" / "configs" / "hpc_smoke_gpu.yaml"
    diag_path = ROOT / "hpc" / "configs" / "hpc_frozen_qwen_diag.yaml"
    main = load_config(main_path)          # raises if invalid
    diag = load_config(diag_path)
    check(True, "both HPC configs parse and pass validate_config")

    # --- 2. scientific conditions of the MAIN smoke config ---
    d = to_plain_dict(main)
    check(main.backbone.backend == "qwen3vl", "main: real qwen3vl backend")
    check(main.backbone.tune_language_model is True,
          "main: language backbone is TRAINED (not frozen)")
    check(main.backbone.tune_qwen_vision is False and main.backbone.freeze_qwen_vision is True,
          "main: Qwen vision encoder frozen (engineering default, unchanged)")
    check(main.backbone.lora_enabled is False, "main: LoRA is OFF")
    check(main.backbone.attn_implementation == "eager",
          "main: attention backend eager (required for 2nd-order backward)")
    check(main.backbone.dtype == "bfloat16", "main: Qwen backbone bf16 (A100 memory)")
    check(main.trainer.candidate_dtype == "float32",
          "main: candidates/energy/loss stay float32")
    check(main.solver.unroll_mode == "full", "main: FULL unroll (no truncation)")
    check(main.solver.joint_update_mode == "simultaneous",
          "main: simultaneous joint update")
    check(main.solver.train_inner_steps == 2 and main.solver.eval_inner_steps == 2,
          "main: K=2 inner steps for both train and eval")
    check(main.trainer.per_device_batch_size == 1, "main: batch size 1")
    check(main.trainer.max_train_steps == 10, "main: bounded to 10 steps")
    # geometry must be the real task geometry, not a shortened one
    check(main.data.action_dim == 12, "main: action_dim 12")
    check(main.data.state_dim == 32, "main: state_dim 32 (sincos of 16)")
    check(main.data.action_horizon == 16, "main: action_horizon 16")
    check(main.data.num_future_frames == 3, "main: 3 future frames")
    check(list(main.data.future_offsets) == [5, 11, 16], "main: future offsets [5,11,16]")
    check(main.data.num_cameras == 1 and list(main.data.camera_order) == ["robot0_agentview_left"],
          "main: single left camera")
    check(main.data.normalization_profile == "paper_standardized_v1",
          "main: paper §4.8 standardized coordinate")
    check(main.data.action_normalization == "zscore", "main: zscore actions")
    check(main.data.standardization_epsilon == 1e-6, "main: epsilon 1e-6")
    check(main.init.initialization_action == "normal" and main.init.init_action_std == 1.0
          and main.init.init_action_mean == 0.0, "main: N(0,I) action init")
    check(main.init.initialization_future == "normal" and main.init.init_future_std == 1.0
          and main.init.init_future_mean == 0.0, "main: N(0,I) future init")
    # optimizer untouched
    check(main.trainer.optimizer == "AdamW" and main.trainer.learning_rate == 1e-5
          and main.trainer.weight_decay == 0.0 and main.trainer.max_grad_norm == 1.0,
          "main: AdamW lr=1e-5 wd=0 clip=1.0 (unchanged)")
    check(main.trainer.device == "cuda", "main: device cuda")

    # --- 3. the diagnostic must differ in EXACTLY one key ---
    dd = to_plain_dict(diag)

    def flatten(node, prefix=""):
        out = {}
        if isinstance(node, dict):
            for k, v in node.items():
                out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[prefix[:-1]] = node
        return out

    fm, fd = flatten(d), flatten(dd)
    check(set(fm) == set(fd), "diag: same config keys as main (no hidden extras)")
    diffs = sorted(k for k in fm if fm[k] != fd.get(k))
    allowed = {"backbone.tune_language_model", "run_id", "notes",
               "trainer.output_dir"}
    unexpected = [k for k in diffs if k not in allowed]
    check(not unexpected, f"diag: only labelled deviations differ (diffs={diffs})")
    check(fd["backbone.tune_language_model"] is False, "diag: language frozen")
    check("NOT" in dd["notes"] and "DIAGNOSTIC" in dd["notes"].upper(),
          "diag: notes explicitly say NOT a scientific result")


# ---------------------------------------------------------------------------
# 4. manifest generator
# ---------------------------------------------------------------------------
def test_manifests() -> None:
    print("[4] deterministic episode-disjoint manifests")
    from synthetic_lerobot import write_synthetic_lerobot

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = write_synthetic_lerobot(td / "data", num_episodes=20, episode_length=12,
                                       dataset_name="fixture_mix")
        out_a = td / "man_a"
        out_b = td / "man_b"
        script = ROOT / "hpc" / "scripts" / "make_episode_manifests.py"

        def run(out_dir: Path, seed: int, extra=()):
            cmd = [PY, str(script), "--dataset-root", str(base), "--out-dir", str(out_dir),
                   "--seed", str(seed), "--val-frac", "0.2", "--test-frac", "0.1",
                   "--overwrite", *extra]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout, r.stderr)
            check(r.returncode == 0, f"generator exits 0 (seed={seed})")

        run(out_a, 20261029)
        run(out_b, 20261029)
        run(td / "man_c", 7)

        tr = json.loads((out_a / "train_episodes.json").read_text())
        va = json.loads((out_a / "val_episodes.json").read_text())
        te = json.loads((out_a / "test_episodes.json").read_text())

        check(tr["schema_version"] == "energywam_episode_manifest_v1",
              "manifest schema_version is energywam_episode_manifest_v1")
        check(tr["split"] == "train" and va["split"] == "val" and te["split"] == "test",
              "splits are labelled train/val/test")
        check(tr["dataset_identity"].endswith("/SyntheticRobot"),
              f"dataset_identity taken from the reader ({tr['dataset_identity']})")
        check(Path(tr["dataset_root"]).resolve() == base.resolve(),
              "dataset_root recorded as the absolute dataset path")

        all_eps = set(tr["episode_indices"]) | set(va["episode_indices"]) | set(te["episode_indices"])
        check(all_eps == set(range(20)), "split is a PARTITION of all 20 episodes")
        check(not (set(tr["episode_indices"]) & set(va["episode_indices"])), "train/val disjoint")
        check(not (set(tr["episode_indices"]) & set(te["episode_indices"])), "train/test disjoint")
        check(not (set(va["episode_indices"]) & set(te["episode_indices"])), "val/test disjoint")
        check(len(tr["episode_indices"]) == 14 and len(va["episode_indices"]) == 4
              and len(te["episode_indices"]) == 2,
              "counts follow the floor rule (20 * 0.8/0.2/0.1 -> 14/4/2)")
        check(all(len(set(m["episode_indices"])) == len(m["episode_indices"])
                  for m in (tr, va, te)), "no duplicates inside a split")

        # determinism: byte-identical output for the same seed
        same = all((out_a / f).read_bytes() == (out_b / f).read_bytes()
                   for f in ("train_episodes.json", "val_episodes.json", "test_episodes.json"))
        check(same, "same seed -> byte-identical manifests")
        diff = (out_a / "train_episodes.json").read_bytes() != \
               (td / "man_c" / "train_episodes.json").read_bytes()
        check(diff, "different seed -> different split")

        # the project's own loader must accept these files, with the reader attached
        from energywam.data.episode_dataset import LeRobotEpisodeReader
        from energywam.data.stats_fit import load_episode_manifest

        reader = LeRobotEpisodeReader(base)
        man, eps = load_episode_manifest(out_a / "train_episodes.json", reader=reader,
                                         expect_split="train",
                                         other_manifest_paths=[out_a / "val_episodes.json",
                                                               out_a / "test_episodes.json"])
        check(len(eps) == 14 and "manifest_sha256" in man,
              "project loader validates the train manifest against the dataset")

        # a val manifest must be refused by the TRAIN-only statistics path
        try:
            load_episode_manifest(out_a / "val_episodes.json", reader=reader, expect_split="train")
            check(False, "val manifest refused by the train-only statistics loader")
        except Exception as exc:
            check("TRAIN split only" in str(exc), "val manifest refused by the train-only statistics loader")

        # overlapping manifests must be refused
        bad = td / "bad_train.json"
        bad.write_text(json.dumps({
            "schema_version": "energywam_episode_manifest_v1", "split": "train",
            "dataset_identity": tr["dataset_identity"], "dataset_root": tr["dataset_root"],
            "episode_indices": tr["episode_indices"] + [va["episode_indices"][0]],
            "notes": "deliberately leaking one val episode",
        }))
        try:
            load_episode_manifest(bad, reader=reader, expect_split="train",
                                  other_manifest_paths=[out_a / "val_episodes.json"])
            check(False, "cross-split leakage is refused")
        except Exception as exc:
            check("appear in both" in str(exc), "cross-split leakage is refused")

        # no synthetic mode: a non-dataset directory is refused
        r = subprocess.run([PY, str(script), "--dataset-root", str(td / "nowhere"),
                            "--out-dir", str(td / "man_x"), "--overwrite"],
                           capture_output=True, text=True)
        check(r.returncode != 0 and "not a LeRobot dataset" in (r.stdout + r.stderr),
              "refuses a directory that is not a LeRobot dataset")


def main() -> int:
    test_configs()
    test_manifests()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for m in FAIL:
        print(f"  FAILED: {m}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
