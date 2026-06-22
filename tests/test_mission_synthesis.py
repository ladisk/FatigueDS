import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from FatigueDS.mission_synthesis import combine_fds, envelope_ers


def test_combine_fds_sums():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 0.5, 0.5])
    assert np.allclose(combine_fds([a, b]), np.array([1.5, 2.5, 3.5]))


def test_combine_fds_repeats():
    a = np.array([1.0, 2.0, 3.0])
    assert np.allclose(combine_fds([a], repeats=[2]), np.array([2.0, 4.0, 6.0]))


def test_combine_fds_length_mismatch():
    with pytest.raises(ValueError):
        combine_fds([np.array([1.0, 2.0]), np.array([1.0])])


def test_combine_fds_bad_repeats():
    with pytest.raises(ValueError):
        combine_fds([np.array([1.0, 2.0])], repeats=[0])


def test_envelope_ers_max():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([4.0, 2.0, 3.0])
    assert np.allclose(envelope_ers([a, b]), np.array([4.0, 5.0, 3.0]))


import FatigueDS
from FatigueDS.mission_synthesis import invert_fds_to_psd


def test_invert_roundtrip_closure():
    """Forward FDS (Spectrum) -> invert -> recover the original flat PSD in-band.
    Wide band + high Q so the narrow-band inverse [4.9] matches the full integral."""
    df = 0.5
    freq = np.arange(10.0, 4000.0 + df, df)
    G = 2.0
    psd = np.full_like(freq, G)
    f0 = np.array([500.0, 1000.0, 1500.0])
    Q, k, C, p, T = 50, 1.0, 1.0, 1.0, 3600.0

    s = FatigueDS.Spectrum(freq_data=f0, Q=Q)
    s.set_random_load((psd, freq), unit='ms2', T=T)
    s.get_fds(k=k, C=C, p=p)

    recovered = invert_fds_to_psd(s.fds, f0, k=k, C=C, p=p, Q=Q, T_test=T)
    assert np.allclose(recovered, G, rtol=0.05)


def test_invert_zero_fds_gives_zero_psd():
    f0 = np.array([100.0, 200.0])
    fds = np.array([0.0, 0.0])
    out = invert_fds_to_psd(fds, f0, k=8, C=1.0, p=1.0, Q=10, T_test=3600.0)
    assert np.allclose(out, 0.0)


def test_invert_acceleration_scaling():
    """Shorter test duration -> higher PSD by factor (T/T_test)**(2/k)."""
    f0 = np.array([300.0, 600.0])
    fds = np.array([1e-20, 2e-20])
    k = 8
    g_full = invert_fds_to_psd(fds, f0, k=k, C=1.0, p=1.0, Q=10, T_test=3600.0)
    g_fast = invert_fds_to_psd(fds, f0, k=k, C=1.0, p=1.0, Q=10, T_test=360.0)
    assert np.allclose(g_fast / g_full, 10 ** (2 / k))
