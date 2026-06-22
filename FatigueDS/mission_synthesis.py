import numpy as np
from scipy.special import gamma


def combine_fds(fds_list, repeats=None):
    """Combine event FDS by damage summation: sum_i repeats_i * fds_i.

    :param fds_list: list of 1-D FDS arrays, all the same length
    :param repeats: optional per-event occurrence multipliers (default all 1)
    :return: combined FDS array
    """
    if len(fds_list) == 0:
        raise ValueError("fds_list must contain at least one FDS array")
    arrs = [np.asarray(f, dtype=float) for f in fds_list]
    n = arrs[0].shape[0]
    if any(a.shape != (n,) for a in arrs):
        raise ValueError("all FDS arrays must be 1-D and the same length")
    if repeats is None:
        repeats = np.ones(len(arrs))
    repeats = np.asarray(repeats, dtype=float)
    if repeats.shape != (len(arrs),):
        raise ValueError("repeats length must match the number of FDS arrays")
    if np.any(repeats <= 0):
        raise ValueError("repeats must be positive")
    total = np.zeros(n)
    for a, r in zip(arrs, repeats):
        total += r * a
    return total


def envelope_ers(ers_list):
    """Combine event ERS by elementwise maximum (extreme-response envelope).

    :param ers_list: list of 1-D ERS arrays, all the same length
    :return: enveloped ERS array
    """
    if len(ers_list) == 0:
        raise ValueError("ers_list must contain at least one ERS array")
    arrs = [np.asarray(e, dtype=float) for e in ers_list]
    n = arrs[0].shape[0]
    if any(a.shape != (n,) for a in arrs):
        raise ValueError("all ERS arrays must be 1-D and the same length")
    return np.max(np.vstack(arrs), axis=0)
