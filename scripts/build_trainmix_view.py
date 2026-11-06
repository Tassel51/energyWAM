#!/usr/bin/env python3 -I
"""Build a merged LeRobot v2.1 dataset VIEW over the 50 RoboCasa365 target tasks.

The energyWAM reader (energywam/data/episode_dataset.py) reads exactly ONE
LeRobot dataset root (meta/info.json + meta/episodes.jsonl + data/ + videos/).
The verified task tree at /datasets/robocasa365/v1.0/target holds 50 separate
datasets.  This script creates a read-only merged *view* via symlinks — no data
is copied and the verified tree is not touched:

    <out_root>/<out_name>/
        meta/info.json         (aggregated totals, templates/features copied)
        meta/modality.json     (copied; asserted identical across tasks)
        meta/stats.json        (copied; loaded unconditionally by the reader)
        meta/episodes.jsonl    (global episode_index, source traceability)
        data/chunk-000/episode_{global:06d}.parquet        -> source parquet
        videos/chunk-000/<CAM>/episode_{global:06d}.mp4    -> source mp4

It also writes deterministic train/val/test episode manifests
(``energywam_episode_manifest_v1``) used by stats fitting and evaluation.

Split rule (fixed, deterministic): within each source task, source episode
index % 20 == 0 -> test, == 1 -> val, else train.

Usage:
    python3 -I scripts/build_trainmix_view.py \
        --src /datasets/robocasa365/v1.0/target \
        --out-root /datasets/robocasa365_trainmix \
        --out-name robocasa365_target_50tasks \
        --manifests /root/zzc/energyWAM/manifests
"""
import argparse
import json
import os
import random
import sys

EXPECTED = [
    ("atomic", "CloseBlenderLid", "20250822"),
    ("atomic", "CloseFridge", "20250816"),
    ("atomic", "CloseToasterOvenDoor", "20250818"),
    ("atomic", "CoffeeSetupMug", "20250813"),
    ("atomic", "NavigateKitchen", "20250821"),
    ("atomic", "OpenCabinet", "20250813"),
    ("atomic", "OpenDrawer", "20250816"),
    ("atomic", "OpenStandMixerHead", "20250818"),
    ("atomic", "PickPlaceCounterToCabinet", "20250811"),
    ("atomic", "PickPlaceCounterToStove", "20250818"),
    ("atomic", "PickPlaceDrawerToCounter", "20250820"),
    ("atomic", "PickPlaceSinkToCounter", "20250813"),
    ("atomic", "PickPlaceToasterToCounter", "20250817"),
    ("atomic", "SlideDishwasherRack", "20250820"),
    ("atomic", "TurnOffStove", "20250812"),
    ("atomic", "TurnOnElectricKettle", "20250817"),
    ("atomic", "TurnOnMicrowave", "20250813"),
    ("atomic", "TurnOnSinkFaucet", "20250812"),
    ("composite", "ArrangeBreadBasket", "20250809"),
    ("composite", "ArrangeTea", "20250812"),
    ("composite", "BreadSelection", "20250815"),
    ("composite", "CategorizeCondiments", "20250814"),
    ("composite", "CuttingToolSelection", "20250814"),
    ("composite", "DeliverStraw", "20250813"),
    ("composite", "GarnishPancake", "20250815"),
    ("composite", "GatherTableware", "20250815"),
    ("composite", "GetToastedBread", "20250812"),
    ("composite", "HeatKebabSandwich", "20250813"),
    ("composite", "KettleBoiling", "20250814"),
    ("composite", "LoadDishwasher", "20250811"),
    ("composite", "MakeIceLemonade", "20250813"),
    ("composite", "PackIdenticalLunches", "20250815"),
    ("composite", "PanTransfer", "20250817"),
    ("composite", "PortionHotDogs", "20250816"),
    ("composite", "PreSoakPan", "20250809"),
    ("composite", "PrepareCoffee", "20250812"),
    ("composite", "RecycleBottlesByType", "20250812"),
    ("composite", "RinseSinkBasin", "20250816"),
    ("composite", "ScrubCuttingBoard", "20250816"),
    ("composite", "SearingMeat", "20250812"),
    ("composite", "SeparateFreezerRack", "20250815"),
    ("composite", "SetUpCuttingStation", "20250817"),
    ("composite", "StackBowlsCabinet", "20250815"),
    ("composite", "SteamInMicrowave", "20250814"),
    ("composite", "StirVegetables", "20250814"),
    ("composite", "StoreLeftoversInBowl", "20250813"),
    ("composite", "WaffleReheat", "20250817"),
    ("composite", "WashFruitColander", "20250811"),
    ("composite", "WashLettuce", "20250814"),
    ("composite", "WeighIngredients", "20250812"),
]

CAMERAS = [
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--shuffle-seed", type=int, default=42)
    args = ap.parse_args()

    out = os.path.join(args.out_root, args.out_name)
    meta = os.path.join(out, "meta")
    data_dir = os.path.join(out, "data", "chunk-000")
    os.makedirs(meta, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    for cam in CAMERAS:
        os.makedirs(os.path.join(out, "videos", "chunk-000", cam), exist_ok=True)
    os.makedirs(args.manifests, exist_ok=True)

    # ---- pass 1: invariants across the 50 task datasets --------------------
    ref_info = ref_modality = ref_stats = None
    tasks = []
    for split, task, date in EXPECTED:
        lb = os.path.join(args.src, split, task, date, "lerobot")
        if not os.path.isdir(lb):
            print(f"FATAL missing dataset {lb}")
            return 1
        with open(os.path.join(lb, "meta", "info.json")) as f:
            info = json.load(f)
        with open(os.path.join(lb, "meta", "modality.json")) as f:
            modality = json.load(f)
        with open(os.path.join(lb, "meta", "stats.json")) as f:
            stats = json.load(f)
        for key in ("codebase_version", "robot_type", "fps", "data_path", "video_path", "features"):
            if ref_info is None:
                pass
            elif info.get(key) != ref_info.get(key):
                print(f"FATAL {split}/{task}/{date}: info[{key}] differs from reference")
                return 1
        if ref_info is None:
            ref_info, ref_modality, ref_stats = info, modality, stats
        if modality != ref_modality:
            print(f"FATAL {split}/{task}/{date}: modality.json differs")
            return 1
        if info.get("robot_type") != "PandaOmron":
            print(f"FATAL {split}/{task}/{date}: robot_type={info.get('robot_type')}")
            return 1
        tasks.append((split, task, date, lb))

    # ---- pass 2: enumerate episodes, assign global indices + splits --------
    records = []  # (global_ep, split_name, src_lb, src_ep, rec)
    total_frames = 0
    for split, task, date, lb in tasks:
        with open(os.path.join(lb, "meta", "episodes.jsonl")) as f:
            eps = [json.loads(line) for line in f if line.strip()]
        for rec in eps:
            src_ep = int(rec["episode_index"])
            g = len(records)
            split_name = "test" if src_ep % 20 == 0 else ("val" if src_ep % 20 == 1 else "train")
            rec = dict(rec)
            rec["episode_index"] = g
            rec["source_dataset"] = f"{split}/{task}/{date}"
            rec["source_episode_index"] = src_ep
            if not isinstance(rec.get("length"), int) or rec["length"] <= 0:
                print(f"FATAL {split}/{task}/{date} ep {src_ep}: bad length")
                return 1
            if not rec.get("tasks"):
                print(f"FATAL {split}/{task}/{date} ep {src_ep}: empty tasks")
                return 1
            total_frames += int(rec["length"])
            records.append((g, split_name, lb, src_ep, rec))

    n = len(records)
    print(f"episodes: {n}, total_frames: {total_frames}")

    # ---- write episodes.jsonl --------------------------------------------
    with open(os.path.join(meta, "episodes.jsonl"), "w") as out_f:
        for _g, _s, _lb, _src, rec in records:
            out_f.write(json.dumps(rec) + "\n")

    # ---- write info.json / modality.json / stats.json ---------------------
    info = dict(ref_info)
    info["total_episodes"] = n
    info["total_frames"] = total_frames
    info["total_videos"] = n * len(CAMERAS)
    info["total_tasks"] = len(tasks)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump(info, f)
    with open(os.path.join(meta, "modality.json"), "w") as f:
        json.dump(ref_modality, f)
    with open(os.path.join(meta, "stats.json"), "w") as f:
        json.dump(ref_stats, f)

    # ---- symlinks ---------------------------------------------------------
    npar = nmp4 = 0
    for g, _s, lb, src_ep, _rec in records:
        src_par = os.path.join(lb, "data", "chunk-000", f"episode_{src_ep:06d}.parquet")
        if not os.path.isfile(src_par):
            print(f"FATAL missing parquet {src_par}")
            return 1
        os.symlink(src_par, os.path.join(data_dir, f"episode_{g:06d}.parquet"))
        npar += 1
        for cam in CAMERAS:
            src_mp4 = os.path.join(lb, "videos", "chunk-000", cam, f"episode_{src_ep:06d}.mp4")
            if not os.path.isfile(src_mp4):
                print(f"FATAL missing mp4 {src_mp4}")
                return 1
            os.symlink(src_mp4, os.path.join(out, "videos", "chunk-000", cam, f"episode_{g:06d}.mp4"))
            nmp4 += 1
    print(f"symlinks: parquet={npar} mp4={nmp4}")

    # ---- manifests --------------------------------------------------------
    identity = f"{args.out_name}/{ref_info.get('robot_type', 'unknown')}"
    by_split = {"train": [], "val": [], "test": []}
    for g, s, _lb, _src, _rec in records:
        by_split[s].append(g)
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(by_split["train"])  # spread --max-samples windows across tasks
    for s in ("train", "val", "test"):
        man = {
            "schema_version": "energywam_episode_manifest_v1",
            "split": s,
            "dataset_identity": identity,
            "dataset_root": out,
            "episode_indices": by_split[s],
            "notes": (
                "merged 50-task target view (symlinks); split rule: per source task, "
                "source_episode_index %% 20: 0->test, 1->val, else train; "
                "train list deterministically shuffled (seed %d)" % args.shuffle_seed
            ),
        }
        path = os.path.join(args.manifests, f"{s}_episodes.json")
        with open(path, "w") as f:
            json.dump(man, f)
        print(f"manifest {s}: {len(by_split[s])} episodes -> {path}")

    print(f"dataset_identity: {identity}")
    print("BUILD-VIEW-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
