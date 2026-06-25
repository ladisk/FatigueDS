import pytest
import numpy as np
import sys
import os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import FatigueDS

from test_data import *

# Pytest will discover and run all test functions named `test_*` or `*_test`.

def test_version():
    """ check sdypy_template_project exposes a version attribute """
    assert hasattr(FatigueDS, "__version__")
    assert isinstance(FatigueDS.__version__, str)


class TestCore:
    """ Testing core functions """

    def test_sine(self):
        """ Test the sine function """
        load_spectrum_sine = FatigueDS.Spectrum(freq_data=(0, 2000, 5))
        load_spectrum_sine.set_sine_load(sine_freq=500, amp=10, t_total=3600)
        load_spectrum_sine.get_ers()
        load_spectrum_sine.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_sine.ers, sine_ers_true)
        assert np.allclose(load_spectrum_sine.fds, sine_fds_true)

    def test_sine_sweep(self):
        """ Test the sine sweep function """
        load_spectrum_sine_sweep = FatigueDS.Spectrum(freq_data=(0, 2000, 5))
        load_spectrum_sine_sweep.set_sine_sweep_load(const_amp=[5,10,20], const_f_range=[20,100,500,1000], exc_type='acc', sweep_type='log', sweep_rate=1)
        load_spectrum_sine_sweep.get_ers()
        load_spectrum_sine_sweep.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_sine_sweep.ers, sine_sweep_ers_true)
        assert np.allclose(load_spectrum_sine_sweep.fds, sine_sweep_fds_true)

    def test_random_psd(self):
        """ Test the random psd function """
        _psd_data = np.load('test_data/test_psd.npy', allow_pickle=True)
        psd_freq = _psd_data[:,0]
        psd_data = _psd_data[:,1]

        load_spectrum_psd = FatigueDS.Spectrum(freq_data=(20, 200, 5))
        load_spectrum_psd.set_random_load((psd_data, psd_freq), unit='g', T=133.5711234541)
        load_spectrum_psd.get_ers()
        load_spectrum_psd.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_psd.ers, random_psd_ers_true)
        assert np.allclose(load_spectrum_psd.fds, random_psd_fds_true)

    def test_random_time_convolution(self):
        """ Test the random time history function with convolution"""
        _time_data = np.load('test_data/test_time_history.npy', allow_pickle=True)
        time_history_data = _time_data[:,1]
        t = _time_data[:,0] 
        dt = t[2] - t[1]

        load_spectrum_convolution = FatigueDS.Spectrum(freq_data=(20, 200, 5))
        load_spectrum_convolution.set_random_load((time_history_data, dt), unit='g')
        load_spectrum_convolution.get_ers()
        load_spectrum_convolution.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_convolution.ers, random_time_convolution_ers_true)
        assert np.allclose(load_spectrum_convolution.fds, random_time_convolution_fds_true)

    def test_random_time_psd_averaging(self):
        """ Test the random time history function with psd averaging"""
        _time_data = np.load('test_data/test_time_history.npy', allow_pickle=True)
        time_history_data = _time_data[:,1]
        t = _time_data[:,0]
        dt = t[2] - t[1]

        load_spectrum_averaging = FatigueDS.Spectrum(freq_data=(20, 200, 5))
        load_spectrum_averaging.set_random_load((time_history_data, dt), unit='g', method='psd_averaging', bins=10)
        load_spectrum_averaging.get_ers()
        load_spectrum_averaging.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_averaging.ers, random_time_averaging_ers_true)
        assert np.allclose(load_spectrum_averaging.fds, random_time_averaging_fds_true)


def test_narrowband_crossing_rate_equals_f0():
    """Absolute, literature-anchored check of the FDS/ERS cycle-rate convention.

    For a single-DOF oscillator (Q=10) excited by broadband (white) random vibration,
    the response is narrow-band and its mean number of upward zero-crossings per
    second equals the natural frequency f0 (Lalanne, *Specification Development*,
    Vol.5 p.46: "n0 is equal to f0"; verified numerically in Vol.5 Example 4.4, where
    a 10 Hz oscillator shows 50 up-crossings in 5 s -> 10 Hz). This is the quantity
    n0+ used in the random ERS/FDS, so it pins down the (1/(2*pi)) factor in
    n0+ = (1/(2*pi)) * (dz_rms/z_rms) and guards against the factor-of-2 error of
    using 1/pi (which would yield ~2*f0).
    """
    rng = np.random.default_rng(0)
    fs = 2048.0
    dt = 1 / fs
    acc = rng.standard_normal(int(120.0 * fs))  # broadband (white) excitation
    damp = 1 / (2 * 10)  # Q = 10

    for f0 in (50.0, 100.0, 150.0, 200.0):
        z = FatigueDS.tools.response_relative_displacement(acc, dt, f_0=f0, damp=damp)
        z = z[len(z) // 20:]  # drop start-up transient
        n_up = np.count_nonzero((z[:-1] < 0) & (z[1:] >= 0))
        rate = n_up / (len(z) * dt)
        assert np.isclose(rate, f0, rtol=0.02), f"f0={f0}: crossing rate {rate:.2f} != f0"


class _SpectralDataLike:
    """Minimal stand-in for ``FLife.SpectralData``: exposes ``.psd`` as the (N, 2)
    [frequency, PSD] array that ``set_random_load`` reads. Lets us test the
    SpectralData input branch without importing FLife (and its GUI dependencies)."""

    def __init__(self, psd):
        self.psd = np.asarray(psd)


def test_set_random_load_accepts_spectraldata():
    """A FLife.SpectralData-like object is accepted by set_random_load and yields the
    same ERS/FDS as the equivalent (psd, freq) tuple input."""
    freq = np.arange(0.0, 2000.0, 1.0)
    psd = np.where((freq >= 50) & (freq <= 800), 2.0, 0.0)
    fr = (20, 1000, 20)

    ref = FatigueDS.Spectrum(freq_data=fr, Q=10)
    ref.set_random_load((psd, freq), unit='ms2', T=100.0)
    ref.get_ers()
    ref.get_fds(k=6, C=1.0, p=1.0)

    via = FatigueDS.Spectrum(freq_data=fr, Q=10)
    via.set_random_load(_SpectralDataLike(np.column_stack((freq, psd))), unit='ms2', T=100.0)
    via.get_ers()
    via.get_fds(k=6, C=1.0, p=1.0)

    assert via.signal_type == 'random_psd'
    assert np.allclose(ref.ers, via.ers)
    assert np.allclose(ref.fds, via.fds)


def test_set_random_load_spectraldata_bad_shape_raises():
    """A SpectralData-like object whose .psd is not (N, 2) is rejected."""
    s = FatigueDS.Spectrum(freq_data=(20, 200, 5), Q=10)
    with pytest.raises(ValueError):
        s.set_random_load(_SpectralDataLike(np.arange(10.0)), unit='ms2', T=100.0)


def test_set_random_load_spectraldata_requires_T():
    """PSD input via a SpectralData still requires the duration T."""
    freq = np.arange(0.0, 500.0, 1.0)
    psd = np.ones_like(freq)
    s = FatigueDS.Spectrum(freq_data=(20, 200, 5), Q=10)
    with pytest.raises(ValueError):
        s.set_random_load(_SpectralDataLike(np.column_stack((freq, psd))), unit='ms2')  # no T


def _basquin_to_sn_reference(sigma_f, b, range=False):
    """Independent oracle for the Basquin -> S-N convention (matches FLife.tools.basquin_to_sn)."""
    k = -1.0 / b
    if not range:
        C = 0.5 * sigma_f ** k
    else:
        C = 0.5 * (2.0 * sigma_f) ** k
    return C, k


@pytest.mark.parametrize("range_flag", [False, True])
def test_material_parameters_convert_to_basquin_roundtrip(range_flag):
    """material_parameters_convert_to_basquin is the exact inverse of the Basquin -> S-N
    convention (FLife-free; an independent oracle supplies the forward direction)."""
    sigma_f, b = 800.0, -0.1
    C, k = _basquin_to_sn_reference(sigma_f, b, range=range_flag)

    sf, bb = FatigueDS.tools.material_parameters_convert_to_basquin(C, k, range=range_flag)
    assert np.isclose(sf, sigma_f)
    assert np.isclose(bb, b)

    # explicit analytic forms
    assert np.isclose(bb, -1.0 / k)
    expected_sf = (2 * C) ** (1 / k) if not range_flag else 0.5 * (2 * C) ** (1 / k)
    assert np.isclose(sf, expected_sf)


