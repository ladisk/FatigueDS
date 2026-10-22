import os
import sys
import pytest
import numpy as np

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import FatigueDS
from test_data import *


def test_version():
    """ check sdypy_template_project exposes a version attribute """
    assert hasattr(FatigueDS, "__version__")
    assert isinstance(FatigueDS.__version__, str)

class TestUnits:
    """ Testing ers and fds funtions unit scaling. 
    
    - ERS: no scaling applied
    - FDS: scaling is applied.
    """

    def test_units_time(self):
        time_data = np.load('test_data/test_time_history.npy', allow_pickle=True)
        time_history_data = time_data[:,1]
        t = time_data[:,0] 
        dt = t[2] - t[1]

        load_spectrum_g = FatigueDS.Spectrum(freq_data=(20, 200, 5))  # time history (psd averaging)
        load_spectrum_ms2 = FatigueDS.Spectrum(freq_data=(20, 200, 5))  # time history (psd averaging)

        load_spectrum_g.set_random_load((time_history_data, dt), unit='g')  # (time history, dt)
        load_spectrum_ms2.set_random_load((time_history_data * 9.81, dt), unit='ms2')  # (time history, dt)

        load_spectrum_g.get_ers()
        load_spectrum_ms2.get_ers()

        load_spectrum_g.get_fds(k=5, C=1, p=1)
        load_spectrum_ms2.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_g.ers * 9.81, load_spectrum_ms2.ers)
        assert np.allclose(load_spectrum_g.fds, load_spectrum_ms2.fds)

    def test_units_psd(self):
        _psd_data = np.load('test_data/test_psd.npy', allow_pickle=True)
        psd_freq = _psd_data[:,0]
        psd_data = _psd_data[:,1]

        load_spectrum_psd_g = FatigueDS.Spectrum(freq_data=(20, 200, 5))
        load_spectrum_psd_ms2 = FatigueDS.Spectrum(freq_data=(20, 200, 5))
        load_spectrum_psd_g.set_random_load((psd_data, psd_freq), unit='g', T=133.5711234541)
        load_spectrum_psd_ms2.set_random_load((psd_data * 9.81**2, psd_freq), unit='ms2', T=133.5711234541)
        load_spectrum_psd_g.get_ers()
        load_spectrum_psd_ms2.get_ers()
        load_spectrum_psd_g.get_fds(k=5, C=1, p=1)
        load_spectrum_psd_ms2.get_fds(k=5, C=1, p=1)

        assert np.allclose(load_spectrum_psd_g.ers * 9.81, load_spectrum_psd_ms2.ers)
        assert np.allclose(load_spectrum_psd_g.fds, load_spectrum_psd_ms2.fds)

    def test_units_sine(self):
        f0 = np.array([50., 150, 200, 300, 900])
        sp_g = FatigueDS.Spectrum(freq_data=f0, Q=10)
        sp_ms2 = FatigueDS.Spectrum(freq_data=f0, Q=10)
        sp_g.set_sine_load(sine_freq=200, amp=2, t_total=100, unit='g')
        sp_ms2.set_sine_load(sine_freq=200, amp=2 * 9.81, t_total=100, unit='ms2')
        sp_g.get_ers()
        sp_ms2.get_ers()
        sp_g.get_fds(k=5, C=1, p=1)
        sp_ms2.get_fds(k=5, C=1, p=1)

        assert np.allclose(sp_g.ers * 9.81, sp_ms2.ers, rtol=1e-12, atol=0)  # holds before the fix too
        assert np.allclose(sp_g.fds, sp_ms2.fds, rtol=1e-12, atol=0)

    def test_units_sine_sweep(self):
        f0 = np.array([15., 50, 150, 300, 900])
        sp_g = FatigueDS.Spectrum(freq_data=f0, Q=10)
        sp_ms2 = FatigueDS.Spectrum(freq_data=f0, Q=10)
        sp_g.set_sine_sweep_load(const_amp=[2, 4], const_f_range=[10, 100, 1000], sweep_type='log', sweep_rate=1, unit='g')
        sp_ms2.set_sine_sweep_load(const_amp=[2 * 9.81, 4 * 9.81], const_f_range=[10, 100, 1000], sweep_type='log', sweep_rate=1, unit='ms2')
        sp_g.get_ers()
        sp_ms2.get_ers()
        sp_g.get_fds(k=5, C=1, p=1)
        sp_ms2.get_fds(k=5, C=1, p=1)

        assert np.allclose(sp_g.ers * 9.81, sp_ms2.ers, rtol=1e-12, atol=0)  # holds before the fix too
        assert np.allclose(sp_g.fds, sp_ms2.fds, rtol=1e-12, atol=0)
