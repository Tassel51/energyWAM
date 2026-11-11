"""Episode-safe future-frame planning.

The engineering requirements are explicit:

* future frames come from the **same episode** — never read past the end into
  the next trajectory;
* the three offsets are ``round(i*H/3), i=1,2,3`` (for H=16: ``[5, 11, 16]``);
* offsets must be positive, strictly increasing, and aligned with the action
  coverage;
* by default only start indices with a full valid horizon are sampled.

``plan_future_frames`` is the single place where that policy lives, and it is
what the dataset and the tests both call.  It returns a validity flag plus the
absolute frame indices and masks, so a caller can either skip the sample
(``require_full_horizon=True``) or keep it with an explicit mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DataConfig, resolve_future_offsets


@dataclass
class FuturePlan:
    valid: bool
    abs_frames: List[int]
    #: 1 where the future frame is a real frame of THIS episode
    future_valid: List[int]
    #: 1 where the action step is a real step of THIS episode
    action_valid: List[int]
    reason: str = ""

    def as_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "abs_frames": list(self.abs_frames),
            "future_valid": list(self.future_valid),
            "action_valid": list(self.action_valid),
            "reason": self.reason,
        }


def plan_future_frames(
    start_index: int,
    episode_length: int,
    action_horizon: int,
    future_offsets: Sequence[int],
    *,
    require_full_horizon: bool = True,
) -> FuturePlan:
    """Decide whether ``start_index`` yields a valid (action, future) pair.

    Action chunk covers ``[start_index, start_index + H - 1]`` (inclusive), and
    the future offsets are relative to ``start_index``: the frame at
    ``start_index + offset`` is the frame the model is asked to predict.  The
    last offset equals ``H``, i.e. the frame **after** the final action step of
    the chunk, which is what the RoboCasa conversion's ``a_t -> obs_t..obs_{t+1}``
    convention implies (see docs/ARCHITECTURE.md, "action/frame alignment").

    Returns a :class:`FuturePlan`.  ``valid`` is False whenever any required
    frame would fall outside ``[0, episode_length)``.
    """
    H = int(action_horizon)
    offs = [int(o) for o in future_offsets]
    if any(o <= 0 for o in offs):
        return FuturePlan(False, [], [], [], f"non-positive offset in {offs}")
    if sorted(offs) != list(offs) or len(set(offs)) != len(offs):
        return FuturePlan(False, [], [], [], f"offsets not strictly increasing: {offs}")
    if max(offs) > H:
        return FuturePlan(False, [], [], [], f"offset {max(offs)} exceeds action horizon {H}")

    action_idx = [start_index + k for k in range(H)]
    future_idx = [start_index + o for o in offs]

    action_valid = [1 if 0 <= i < episode_length else 0 for i in action_idx]
    future_valid = [1 if 0 <= i < episode_length else 0 for i in future_idx]

    if require_full_horizon:
        ok = all(action_valid) and all(future_valid)
    else:
        ok = all(future_valid)  # at least the supervision targets must exist

    reason = "" if ok else (
        f"start={start_index}, L={episode_length}, H={H}, offs={offs}: "
        f"action_idx={action_idx}, future_idx={future_idx}"
    )
    return FuturePlan(
        valid=ok,
        abs_frames=future_idx,
        future_valid=future_valid,
        action_valid=action_valid,
        reason=reason,
    )


def sample_valid_starts(
    episode_length: int,
    action_horizon: int,
    future_offsets: Sequence[int],
    *,
    require_full_horizon: bool = True,
) -> List[int]:
    """All start indices of an episode that yield a valid plan."""
    out = []
    for t in range(episode_length):
        plan = plan_future_frames(
            t, episode_length, action_horizon, future_offsets,
            require_full_horizon=require_full_horizon,
        )
        if plan.valid:
            out.append(t)
    return out


def offsets_from_config(data_cfg: DataConfig) -> List[int]:
    return resolve_future_offsets(data_cfg)
