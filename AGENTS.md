# Project checkpoint policy

Read `MEMORY.md` at the start of future work in this repository; it records explicit persistent user preferences.

## Experiment logging policy (explicit user instruction, 2026-11-08)

- Keep detailed, durable logs for every experiment, including failed, interrupted, and rejected trials. Chat summaries and W&B alone are insufficient; retain local raw records in the experiment directory.
- Record the purpose, comparison protocol, full resolved hyperparameters, code identity/source snapshot, seeds, data/split/window manifests, initialization/resume source, actual GPU placement, commands, and start/end times with timezone.
- Preserve per-update Action/Future metrics separately, actual optimizer learning rates, completed update counts, sample visits, timings and available resource/gradient diagnostics. Keep step-0 baselines and fixed TRAIN/VAL evaluations distinct from changing training batches; record evaluation cadence and seeds. Mark missing data explicitly; never fabricate, interpolate as measured, or label planned work completed.
- Record latest/best checkpoint steps, selection criterion, resume capability and verification evidence; preserve failures, exit reasons, pauses/resumes, adaptive choices, comparisons and any authorized deletion manifests. Detailed logging does not authorize extra training, evaluation, process restarts, or changes to the agreed protocol.
- Maintain the existing experiment records and canonical handoff rather than creating competing latest reports. Continuous assistant monitoring/chat updates are not required: the user explicitly asked to let the existing automatic queue run and check progress when asked.

User instruction, 2026-11-05: retain the latest checkpoint and the best checkpoint; delete superseded versions.

- For each retained training run, preserve `ckpt_latest.pt` and `ckpt_best.pt`, with metadata. Both should support full resume unless the user explicitly agrees to model-only retention. If best and latest are the same step, aliases/hard links can share one physical snapshot.
- Define the validation criterion for best before launching a run; log Action and Future metrics separately. Do not silently treat the last step or lowest training loss as best.
- Write and verify a new snapshot before atomically replacing the previous corresponding snapshot. Remove older versions only after the latest/best copies are safe.
- Never automatically purge the latest or best, datasets, or pretrained Qwen/DINO weights. A later explicit user instruction may authorize deletion and takes precedence.
- Keep configs, manifests, source snapshots, logs and evaluation reports when removing weight files. Record a deletion manifest and check actual free space.
- `scripts/train_shared_sharded.py --keep-best` now uses `best_checkpoint.py` plus verified rolling latest. A real 4B smoke saved step 1, resumed to step 2, and verified full latest(step 2)/best(step 0) states on 2026-11-05; evidence: `experiments/batch_grid_micro4_20261105/fp32_full_resume_retention_smoke/resume_smoke_result.json` on cloud. This does not validate every other entrypoint or batch configuration. New long runs must explicitly enable best retention and recompute space for both snapshots plus a pending write.
- The user-authorized cloud cleanup is historical: `/cloud/zzc/energyWAM/experiments/checkpoint_cleanup_20261105.json`. Do not rerun this blanket purge for future experiments.

- Explicit user exception, 2026-11-05: passed resume-test weights and clearly rejected short-trial weights may be removed, retaining configs/logs/reports/metadata and a deletion manifest. The four main batch arms and final winning runs still retain full latest/best. The passed two-update resume smoke snapshots have been retired under this exception; see its `retired_full_snapshots.json`. Do not mistake its historical passing report for currently available weights.
