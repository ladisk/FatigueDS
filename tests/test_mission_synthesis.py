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


from FatigueDS.mission_synthesis import MissionSynthesis


def _make_event(freq_range=(20, 200, 5), amp=10.0):
    s = FatigueDS.Spectrum(freq_data=freq_range)
    s.set_sine_load(sine_freq=100, amp=amp, t_total=3600)
    s.get_ers()
    s.get_fds(k=5, C=1, p=1)
    return s


def test_combine_two_events_sum_and_envelope():
    e1 = _make_event(amp=10.0)
    e2 = _make_event(amp=10.0)
    ms = MissionSynthesis([e1, e2])
    ms.combine()
    assert np.allclose(ms.fds_ref, 2 * e1.fds)
    assert np.allclose(ms.ers_ref, np.maximum(e1.ers, e2.ers))
    assert np.allclose(ms.f0_range, e1.f0_range)


def test_combine_repeats():
    e1 = _make_event(amp=10.0)
    ms = MissionSynthesis()
    ms.add_event(e1, repeats=3)
    ms.combine()
    assert np.allclose(ms.fds_ref, 3 * e1.fds)


def test_combine_f0_mismatch_raises():
    e1 = _make_event(freq_range=(20, 200, 5))
    e2 = _make_event(freq_range=(20, 300, 5))
    ms = MissionSynthesis([e1, e2])
    with pytest.raises(ValueError):
        ms.combine()


def test_add_event_bad_repeats():
    ms = MissionSynthesis()
    with pytest.raises(ValueError):
        ms.add_event(_make_event(), repeats=0)


def test_combine_missing_fds_raises():
    s = FatigueDS.Spectrum(freq_data=(20, 200, 5))
    s.set_sine_load(sine_freq=100, amp=10.0, t_total=3600)
    s.get_ers()  # no get_fds -> no .fds
    ms = MissionSynthesis([s])
    with pytest.raises(ValueError):
        ms.combine()
