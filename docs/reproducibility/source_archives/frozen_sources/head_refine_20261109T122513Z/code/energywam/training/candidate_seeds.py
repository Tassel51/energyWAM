"""Reproducible candidate seeds for the simplest fixed-input fitting check."""
import hashlib


def fixed_candidate_seed(seed, role, episode_index, frame_index):
    key=f'{int(seed)}:{int(role)}:{int(episode_index)}:{int(frame_index)}'.encode()
    return int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little') % (2**63)
