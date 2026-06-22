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


