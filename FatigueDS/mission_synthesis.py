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


def invert_fds_to_psd(fds_ref, f0_range, k, C, p, Q, T_test):
    """Invert a reference FDS to an equivalent test acceleration PSD.

    Closed-form inverse of Lalanne (Specification Development, Vol.5) eq [4.9]:

        G(f0) = (2*w0**3 / Q) * [ C*fds_ref / (p**k * f0 * T_test * gamma(1+k/2)) ]**(2/k)

    with w0 = 2*pi*f0 and the narrow-band assumption n0+ = f0. Returns 0 where the
    reference damage is 0 or f0 is 0.

    :param fds_ref: reference fatigue damage spectrum (per f0)
    :param f0_range: natural-frequency axis [Hz]
    :param k: S-N slope (Basquin N*s**k = C)
    :param C: S-N constant
    :param p: stress/relative-displacement proportionality constant
    :param Q: quality factor
    :param T_test: test duration [s]
    :return: acceleration PSD G(f0) on f0_range [(m/s^2)^2/Hz]
    """
    fds_ref = np.asarray(fds_ref, dtype=float)
    f0 = np.asarray(f0_range, dtype=float)
    if T_test <= 0:
        raise ValueError("T_test must be positive")
    w0 = 2 * np.pi * f0
    psd = np.zeros_like(f0)
    mask = (fds_ref > 0) & (f0 > 0)
    inner = C * fds_ref[mask] / (p**k * f0[mask] * T_test * gamma(1 + k / 2))
    psd[mask] = (2 * w0[mask]**3 / Q) * inner**(2 / k)
    return psd
