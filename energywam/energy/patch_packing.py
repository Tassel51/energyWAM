"""Spatial patch rearrangement before projection, without target pooling."""
import math


def pack_patches(value, factor):
    """[..., P, D] -> [..., P/f², f²*D], coarse-grid row-major.

    Values within each coarse patch are ordered small-row, small-column,
    channel. This operation is a permutation, not averaging. The subsequent
    learned projector is a different model parameterization.
    """
    if factor == 1:
        return value
    patches, channels = value.shape[-2:]
    side = math.isqrt(patches)
    if side * side != patches or side % factor:
        raise ValueError('patch packing requires a square grid divisible by factor')
    leading = value.shape[:-2]
    n = len(leading)
    grid = value.reshape(*leading, side//factor, factor, side//factor, factor, channels)
    grouped = grid.permute(*range(n), n, n+2, n+1, n+3, n+4)
    return grouped.reshape(*leading, patches//(factor*factor), factor*factor*channels)
