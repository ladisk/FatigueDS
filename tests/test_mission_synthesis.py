import os
import sys
import warnings

import numpy as np
import pytest

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import FatigueDS
from FatigueDS import MissionSynthesis

F0 = (20, 600, 5)
F0_RANGE = np.arange(20, 605, 5.0)
G = 0.5 * np.exp(-0.5 * ((F0_RANGE - 250) / 80)**2) + 0.05  # (m/s^2)^2/Hz, with a floor
T = 3600
DATA = os.path.join(my_path, '..', 'test_data', 'lalanne_ex11_2_{}.csv')  # digitized [Lalanne5] Example 11.2


def _event(psd, freq, T, k, unit='ms2', f0=F0, Q=10):
    s = FatigueDS.Spectrum(freq_data=f0, Q=Q)
    s.set_random_load((psd, freq), T=T, unit=unit)
    s.get_ers()
    s.get_fds(k=k, C=1, p=1)
    return s


def _psd_spectrum(psd, T, k, f0=F0_RANGE):
    """Independent ERS and FDS of a PSD defined on ``f0``."""
    s = FatigueDS.Spectrum(freq_data=f0.copy(), Q=10)
    s.set_random_load((psd, f0.copy()), T=T)
    s.get_ers()
    s.get_fds(k=k)
    return s


@pytest.mark.parametrize('k', [4, 8])
@pytest.mark.parametrize('T_test', [T, T / 10])
def test_invert_recovers_psd(k, T_test):
    ms = MissionSynthesis([_event(G, F0_RANGE, T, k)]).invert(T_test, tol=1e-3)

    # time reduction: same damage with PSD * (T/T_test)**(2/b), [Lalanne5] p.124 eq [4.14]
    interior = slice(2, -2)
    expected = G * (T / T_test)**(2 / k)
    np.testing.assert_allclose(ms.test_psd[interior], expected[interior], rtol=1e-2, atol=0)
    np.testing.assert_allclose(ms.test_psd, expected, rtol=5e-2, atol=0)

    fds = _psd_spectrum(ms.test_psd, T_test, k).fds
    np.testing.assert_allclose(fds, ms.fds_ref, rtol=1e-3, atol=0)


def test_invert_single_step_is_eq_11_11():
    k = 8
    ms = MissionSynthesis([_event(G, F0_RANGE, T, k)])
    with pytest.warns(UserWarning, match='FDS'):
        ms.invert(T / 10, max_iter=1)

    # [Lalanne5] p.313 eq [11.11], from the constant 0.1 start PSD
    start = np.full_like(F0_RANGE, 0.1)
    expected = start * (ms.fds_ref / _psd_spectrum(start, T / 10, k).fds)**(2 / k)
    np.testing.assert_allclose(ms.test_psd, expected, rtol=1e-12, atol=0)
    assert ms.invert_n_iter == 1

    # fds_test and invert_error belong to the returned test_psd
    np.testing.assert_allclose(ms.fds_test, _psd_spectrum(ms.test_psd, T / 10, k).fds, rtol=1e-12, atol=0)
    assert ms.invert_error == np.max(np.abs(ms.fds_test / ms.fds_ref - 1))


def test_combine_sum_and_envelope():
    low = _event(0.5 * np.exp(-0.5 * ((F0_RANGE - 100) / 40)**2) + 0.01, F0_RANGE, 3 * T, k=6)
    high = _event(1.0 * np.exp(-0.5 * ((F0_RANGE - 450) / 40)**2) + 0.01, F0_RANGE, T, k=6)
    ms = MissionSynthesis([low]).combine()
    ms.add_event(high).add_event(high).invert(T, max_iter=5, tol=1)  # invert combines again

    np.testing.assert_allclose(ms.fds_ref, low.fds + 2 * high.fds, rtol=1e-12, atol=0)
    np.testing.assert_allclose(ms.ers_ref, np.maximum(low.ers, high.ers), rtol=1e-12, atol=0)
    assert np.any(low.ers > high.ers) and np.any(high.ers > low.ers)


def _sine_event_g():
    s = FatigueDS.Spectrum(freq_data=F0, Q=10)
    s.set_sine_load(sine_freq=200, amp=1, t_total=T, unit='g')
    s.get_ers()
    s.get_fds(k=6)
    return s


def _base():
    return _event(G, F0_RANGE, T, 6)


@pytest.mark.parametrize('events, match', [
    (lambda: [_base(), _event(G, F0_RANGE, T, 6, f0=np.arange(21, 606, 5.0))], 'f0_range'),
    (lambda: [_base(), _event(G, F0_RANGE, T, 6, f0=(20, 700, 5))], 'f0_range'),
    (lambda: [_base(), _event(G, F0_RANGE, T, 7)], 'k, C, p'),
    (lambda: [_base(), _event(G, F0_RANGE, T, 6, Q=20)], 'k, C, p'),
    (lambda: [_base(), _event(G, F0_RANGE, T, 6, unit='g')], 'k, C, p'),
    (lambda: [_event(G, F0_RANGE, T, 6, f0=np.geomspace(20, 600, 60))], 'uniformly spaced'),
    (lambda: [_event(G, F0_RANGE, T, 6, f0=(0, 600, 5))], 'above 0 Hz'),
    (lambda: [_event(0 * G, F0_RANGE, T, 6)], 'positive'),
    (lambda: [_sine_event_g()], 'sine'),
    (lambda: [], 'no events'),
])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')  # ERS of a zero PSD is 0/0
def test_combine_raises(events, match):
    with pytest.raises(ValueError, match=match):
        MissionSynthesis(events()).combine()


def test_g_and_ms2_give_same_result():
    k = 8
    ms_ms2 = MissionSynthesis([_event(G, F0_RANGE, T, k)]).invert(T / 10, tol=1e-3).check_ers()
    ms_g = MissionSynthesis([_event(G / 9.81**2, F0_RANGE, T, k, unit='g')]).invert(T / 10, tol=1e-3).check_ers()

    np.testing.assert_allclose(ms_g.test_psd * 9.81**2, ms_ms2.test_psd, rtol=1e-9, atol=0)
    np.testing.assert_allclose(ms_g.ers_ratio, ms_ms2.ers_ratio, rtol=1e-9, atol=0)


def test_check_ers_ratio():
    k = 8
    event = _event(G, F0_RANGE, T, k)
    ms = MissionSynthesis([event]).invert(T / 10, tol=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ms.check_ers()

    expected = _psd_spectrum(G * 10**(2 / k), T / 10, k).ers / event.ers
    np.testing.assert_allclose(ms.ers_ratio, expected, rtol=1e-3, atol=0)
    assert np.all(ms.ers_ratio > 1)  # [Lalanne5] p.314, Fig. 11.19: expected after a time reduction


def test_check_ers_warns():
    ms = MissionSynthesis([_event(G, F0_RANGE, T, 8)])

    ms.invert(10 * T, tol=1e-3)
    with pytest.warns(UserWarning, match='too long'):
        ms.check_ers()

    ms.invert(T / 10, tol=1e-3)
    with pytest.warns(UserWarning, match='SRS'):
        ms.check_ers(srs=1.05 * ms.ers_ref)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ms.check_ers(srs=10 * ms.ers_ref)

    # get_ers gives NaN where n0*T < 1 (a 0.04 s event, f0 = 20 Hz): the check skips it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ms.add_event(_event(G, F0_RANGE, 0.04, 8)).invert(10 * T, tol=1e-3)
    assert np.isnan(ms.ers_ref[0])
    with pytest.warns(UserWarning, match='too long'):
        ms.check_ers()


def test_example_11_2():
    """[Lalanne5] p.316-319, Example 11.2: truck 20 h and missile flight 5 min, Q=10, b=8,
    default ``tol`` and ``max_iter``. PSDs of Figs. 11.23-11.24 scaled to the printed rms."""
    events = []
    for name, rms, T_event in [('truck', 3.024746, 20 * 3600), ('missile', 27.59684, 5 * 60)]:
        f, psd = np.loadtxt(DATA.format('psd_' + name), delimiter=',').T
        freq = np.arange(np.ceil(f[0]), f[-1], 1.0)
        psd = np.interp(freq, f, psd)
        events.append(_event(psd * rms**2 / np.trapezoid(psd, freq), freq, T_event, k=8, f0=(2, 1047, 5)))
    ms = MissionSynthesis(events).combine()

    # reference ERS (envelope) against Fig. 11.26: median ratio 1.007, peak 1076 vs 1052 m/s^2
    f, ers_book, spec_book = np.loadtxt(DATA.format('ers_5h'), delimiter=',').T
    assert abs(np.median(np.interp(f, ms.f0_range, ms.ers_ref) / ers_book) - 1) < 0.03
    i = np.argmax(ms.ers_ref)
    assert abs(ms.f0_range[i] - f[np.argmax(ers_book)]) <= 5  # 172 vs 171.5 Hz
    assert abs(ms.ers_ref[i] / ers_book.max() - 1) < 0.05

    # 5 h specification: test ERS about 0.70 of the reference (Fig. 11.26), test too long (Fig. 11.22)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ms.invert(5 * 3600)
    with pytest.warns(UserWarning, match='too long'):
        ms.check_ers()
    assert 0.65 < np.min(ms.ers_ratio) < 0.75
    peak_ratio = ms.ers_test.max() / ms.ers_ref.max()  # 0.70, book 743/1052 = 0.71
    assert abs(peak_ratio - spec_book.max() / ers_book.max()) < 0.03

    # 5 min specification: the ERSs are very close above 25 Hz (Fig. 11.28)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ms.invert(5 * 60).check_ers()
    np.testing.assert_allclose(ms.ers_ratio[ms.f0_range > 25], 1, rtol=0.03, atol=0)


def test_invert_converges_without_floor():
    """Default ``tol`` and ``max_iter`` for a PSD that rolls off without a floor
    (slowest case measured: largest ``k``)."""
    freq = np.arange(20, 600, 1.0)
    engine = _event(2.0 * np.exp(-0.5 * ((freq - 275) / 70)**2), freq, T, k=12)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ms = MissionSynthesis([engine]).invert(T / 2)
    assert ms.invert_error < 0.05
