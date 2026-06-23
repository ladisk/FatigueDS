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


def test_invert_uses_event_params_and_matches_helper():
    e1 = _make_event(amp=10.0)  # built with k=5, C=1, p=1, Q=10 (default)
    ms = MissionSynthesis([e1])
    ms.combine()
    ms.invert(T_test=1800.0, method='closed_form')
    expected = invert_fds_to_psd(ms.fds_ref, ms.f0_range, k=5, C=1, p=1, Q=10, T_test=1800.0)
    assert np.allclose(ms.test_psd, expected)
    assert np.allclose(ms.test_psd_freq, ms.f0_range)


def test_invert_before_combine_raises():
    ms = MissionSynthesis([_make_event()])
    with pytest.raises(ValueError):
        ms.invert(T_test=1800.0)


def test_invert_bad_T_test_raises():
    ms = MissionSynthesis([_make_event()])
    ms.combine()
    with pytest.raises(ValueError):
        ms.invert(T_test=0)


def test_invert_inconsistent_params_raises():
    e1 = _make_event(amp=10.0)
    e2 = _make_event(amp=10.0)
    e2.k = 7  # force inconsistent S-N slope across events
    ms = MissionSynthesis([e1, e2])
    ms.combine()
    with pytest.raises(ValueError):
        ms.invert(T_test=1800.0)


def test_invert_override_param():
    e1 = _make_event(amp=10.0)
    e2 = _make_event(amp=10.0)
    e2.k = 7
    ms = MissionSynthesis([e1, e2])
    ms.combine()
    ms.invert(T_test=1800.0, k=5, method='closed_form')  # override resolves the inconsistency
    assert ms.k == 5


def _random_event(T):
    rng = np.random.default_rng(0)
    fs = 2048.0
    acc = rng.standard_normal(int(T * fs))
    s = FatigueDS.Spectrum(freq_data=(50, 400, 10), Q=10)
    s.set_random_load((acc, 1 / fs), unit='ms2')
    s.get_ers()
    s.get_fds(k=8, C=1, p=1)
    return s


def test_check_ers_sets_ratio():
    ms = MissionSynthesis([_random_event(T=20.0)])
    ms.combine()
    ms.invert(T_test=20.0, method='closed_form')  # ERS-guard test
    ms.check_ers()
    assert ms.ers_ratio.shape == ms.f0_range.shape
    assert np.all(np.isfinite(ms.ers_ratio))


def test_check_ers_warns_on_overtest():
    ms = MissionSynthesis([_random_event(T=20.0)])
    ms.combine()
    ms.invert(T_test=2.0, method='closed_form')  # 10x acceleration -> higher test ERS
    with pytest.warns(UserWarning):
        ms.check_ers()


def test_check_ers_before_invert_raises():
    ms = MissionSynthesis([_random_event(T=20.0)])
    ms.combine()
    with pytest.raises(ValueError):
        ms.check_ers()


import matplotlib
matplotlib.use('Agg')


def test_exports_from_package():
    assert hasattr(FatigueDS, 'MissionSynthesis')
    assert hasattr(FatigueDS, 'invert_fds_to_psd')
    assert hasattr(FatigueDS, 'invert_fds_to_psd_iterative')


def _psd_event(freq_range=(20, 600, 5), k=8):
    """A Spectrum event whose FDS comes from a known stepped PSD (so we can
    check that an inversion reproduces that FDS)."""
    s = FatigueDS.Spectrum(freq_data=freq_range, Q=10)
    pf = np.arange(20.0, 600.0, 5.0)
    psd = np.where(pf < 150, 0.5, np.where(pf < 350, 3.0, 0.8))
    s.set_random_load((psd, pf), unit='ms2', T=3600.0)
    s.get_ers()
    s.get_fds(k=k, C=1, p=1)
    return s


def _fds_closure_error(ms):
    """max relative error between the FDS of the derived test PSD and fds_ref."""
    s = FatigueDS.Spectrum(freq_data=ms.f0_range, Q=ms.Q)
    s.set_random_load((ms.test_psd, ms.test_psd_freq), unit='ms2', T=ms.T_test)
    s.get_fds(k=ms.k, C=ms.C, p=ms.p)
    m = ms.fds_ref > 0
    return np.max(np.abs(s.fds[m] / ms.fds_ref[m] - 1.0))


def test_invert_iteration_closes_fds():
    """The iterated test PSD reproduces the reference FDS through the full forward
    response (Lalanne eq [11.11]); closure error should be small."""
    ms = MissionSynthesis([_psd_event()])
    ms.combine()
    ms.invert(3600.0, method='iteration', max_iter=300)
    # single-event reference IS the FDS of a PSD, so it is reproducible to high accuracy
    assert _fds_closure_error(ms) < 0.03            # within 3%
    assert ms.invert_n_iter >= 1
    assert ms.invert_error <= 0.03


def test_invert_iteration_beats_closed_form():
    """For a stepped multi-band PSD, the diagonal closed form has large closure
    error (off-resonance coupling ignored); the iteration is far better."""
    ms_i = MissionSynthesis([_psd_event()]); ms_i.combine(); ms_i.invert(3600.0, method='iteration', max_iter=300)
    ms_c = MissionSynthesis([_psd_event()]); ms_c.combine(); ms_c.invert(3600.0, method='closed_form')
    err_iter = _fds_closure_error(ms_i)
    err_closed = _fds_closure_error(ms_c)
    assert err_closed > 0.2                          # diagonal is poor here (>20%)
    assert err_iter < err_closed / 5                 # iteration is much better


def test_invert_closed_form_matches_helper():
    ms = MissionSynthesis([_psd_event()])
    ms.combine()
    ms.invert(3600.0, method='closed_form')
    expected = invert_fds_to_psd(ms.fds_ref, ms.f0_range, k=ms.k, C=ms.C, p=ms.p, Q=ms.Q, T_test=3600.0)
    assert np.allclose(ms.test_psd, expected)


def test_invert_bad_method_raises():
    ms = MissionSynthesis([_psd_event()])
    ms.combine()
    with pytest.raises(ValueError):
        ms.invert(3600.0, method='nonsense')


def test_plots_run():
    ms = MissionSynthesis([_make_event()])
    ms.combine()
    ms.invert(T_test=1800.0, method='closed_form')
    ms.plot_fds()
    ms.plot_ers()
    ms.plot_test_psd()


def test_plot_before_combine_raises():
    ms = MissionSynthesis([_make_event()])
    with pytest.raises(ValueError):
        ms.plot_fds()
