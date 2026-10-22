import pytest
import numpy as np
import scipy.integrate
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
        assert np.allclose(load_spectrum_sine_sweep.fds, sine_sweep_fds_true, rtol=1e-9, atol=0)

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




def _fds_time_integral(f0, amps, f_range, sweep_type, rate, Q, k, a=0, scale=1):
    """Reference sine sweep FDS: [Lalanne5] eq. [3.38], integrated in time with ``quad``.

    D = 1/C * int_0^tb f(t) |l_m|**k / [(1-(f/f0)**2)**2 + (f/(Q*f0))**2]**(k/2) dt, with
    |l_m| = E*Omega**a / omega_0**2 (C = p = 1). It does not use the h substitution or M(h).
    """
    F1 = f_range[0]
    if sweep_type == 'lin':
        alpha = rate / 60  # [Hz/s]
        f = lambda t: F1 + alpha * t
        t_of_f = lambda x: (x - F1) / alpha
    else:
        T1 = 60 / (rate * np.log(2))  # f(t) = F1 * exp(t/T1)
        f = lambda t: F1 * np.exp(t / T1)
        t_of_f = lambda x: T1 * np.log(x / F1)

    def integrand(t, E):
        ft = f(t)
        h = ft / f0
        return ft * (E * scale * (2 * np.pi * ft)**a / (2 * np.pi * f0)**2)**k / ((1 - h**2)**2 + (h / Q)**2)**(k / 2)

    D = 0
    for E, fa, fb in zip(amps, f_range[:-1], f_range[1:]):
        ta, tb = t_of_f(fa), t_of_f(fb)
        # break points at the resonance and at f0*(1 +- m/Q), m = 1, 10
        pts = [t_of_f(f0 * (1 + s * m / Q)) for m in (1, 10) for s in (-1, 1) if fa < f0 * (1 + s * m / Q) < fb]
        pts += [t_of_f(f0)] if fa < f0 < fb else []
        e = [ta] + sorted(pts) + [tb]
        D += sum(scipy.integrate.quad(integrand, lo, hi, args=(E,), epsabs=0, epsrel=1e-12, limit=500)[0] for lo, hi in zip(e[:-1], e[1:]))
    return D


class TestSineSweepFds:
    """ Sine sweep FDS against the time integral [Lalanne5] eq. [3.38] (rtol only). """

    f0_range = np.array([7., 15, 33, 47, 80, 130, 210, 333, 600, 1200])

    @pytest.mark.parametrize('sweep_type, f_range, amps, rate', [
        ('log', [10, 40, 160], [5, 10], 1),
        ('lin', [10, 40, 110], [5, 10], 60),
        ('log', [10, 160], [5], 1),
    ])
    @pytest.mark.parametrize('Q', [10, 200])
    @pytest.mark.parametrize('exc_type, a', [('acc', 0), ('vel', 1)])
    def test_sweep_fds_vs_time_integral(self, sweep_type, f_range, amps, rate, Q, exc_type, a):
        s = FatigueDS.Spectrum(freq_data=self.f0_range, Q=Q)
        s.set_sine_sweep_load(const_amp=amps, const_f_range=f_range, exc_type=exc_type, sweep_type=sweep_type, sweep_rate=rate)
        s.get_fds(k=8, C=1, p=1)
        ref = np.array([_fds_time_integral(f0, amps, f_range, sweep_type, rate, Q, 8, a) for f0 in self.f0_range])
        assert np.allclose(s.fds, ref, rtol=1e-6, atol=0)

    def test_sweep_fds_resonance_near_band_edge(self):
        """ High Q, large k, f0 just above the lower band edge: peak is at the start of a long interval. """
        f0 = np.array([10.5, 21.])
        for f_range, amps, rate in [([10, 160], [5], 1), ([20, 1000], [5], 1)]:
            s = FatigueDS.Spectrum(freq_data=f0, Q=200)
            s.set_sine_sweep_load(const_amp=amps, const_f_range=f_range, sweep_type='log', sweep_rate=rate)
            s.get_fds(k=10, C=1, p=1)
            ref = np.array([_fds_time_integral(f, amps, f_range, 'log', rate, 200, 10) for f in f0])
            assert np.allclose(s.fds, ref, rtol=1e-6, atol=0)

    @pytest.mark.parametrize('sweep_type, rate', [('log', 1), ('lin', 60)])
    def test_sweep_fds_equal_levels_equal_single(self, sweep_type, rate):
        """ Splitting a sweep into levels with equal amplitude must not change the FDS. """
        f0 = np.array([7., 15, 33, 47, 80, 130, 210, 600])
        s_multi = FatigueDS.Spectrum(freq_data=f0, Q=10)
        s_multi.set_sine_sweep_load(const_amp=[5, 5], const_f_range=[10, 40, 160], sweep_type=sweep_type, sweep_rate=rate)
        s_single = FatigueDS.Spectrum(freq_data=f0, Q=10)
        s_single.set_sine_sweep_load(const_amp=[5], const_f_range=[10, 160], sweep_type=sweep_type, sweep_rate=rate)
        s_multi.get_fds(k=5, C=1, p=1)
        s_single.get_fds(k=5, C=1, p=1)
        assert np.allclose(s_multi.fds, s_single.fds, rtol=1e-9, atol=0)


@pytest.mark.parametrize('exc_type, a', [('vel', 1), ('disp', 2)])
def test_sine_ers_vel_disp_equals_acc(exc_type, a):
    """ [Lalanne5] eq. [1.11]: R = E_m * Omega**a / |H|, Omega is the excitation pulsation. """
    sine_freq = 200.
    f0 = np.array([50., 150, 200, 300, 900])
    s_a = FatigueDS.Spectrum(freq_data=f0, Q=10)
    s_a.set_sine_load(sine_freq=sine_freq, amp=1.5, t_total=100, exc_type=exc_type)
    s_acc = FatigueDS.Spectrum(freq_data=f0, Q=10)
    s_acc.set_sine_load(sine_freq=sine_freq, amp=1.5 * (2 * np.pi * sine_freq)**a, t_total=100, exc_type='acc')
    s_a.get_ers()
    s_acc.get_ers()
    assert np.allclose(s_a.ers, s_acc.ers, rtol=1e-12, atol=0)
