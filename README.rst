FatigueDS
-----------------------

Calculating Extreme Response Spectrum (ERS) and Fatigue Damage Spectrum (FDS) of signals. 
Calculations are supported for sine, sine-sweep, and random signals (defined using PSD or time history).
The underlying theory is based on [1].

See the `documentation <https://fatigueds.readthedocs.io/en/latest/index.html>`_ for more information.


Installation
------------------

Use `pip` to install it by:

.. code-block:: console

    $ pip install FatigueDS

Usage
------------------
Some short examples of how to use the package are given below for different types of signals.

Random signals (PSD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of determining the ERS and FDS of a random signal, defined in the frequency domain (PSD):

Generate sample signal PSD:

.. code-block:: python

    import numpy as np
    import pyExSi as es

    # generate random signal
    fs = 5000  # sampling frequency [Hz]
    time = 1  # time duration [s]
    freq_flat = np.arange(0, fs / 2, 1 / time)  # frequency vector
    freq_lower = 200  # PSD lower frequency limit  [Hz]
    freq_upper = 1000  # PSD upper frequency limit [Hz]
    PSD_flat = es.get_psd(freq_flat, freq_lower, freq_upper, variance=800)  # one-sided flat-shaped PSD

Use the package:

.. code-block:: python
    
    import FatigueDS

    # instantiate the Spectrum class 
    # set the frequency range (start, stop, step) and damping ratio
    load_spectrum = FatigueDS.Spectrum(freq_data=(100, 1100, 20), damp=0.05)

    # set the random load
    load_spectrum.set_random_load((PSD_flat, freq_flat), unit='ms2', T=3600)  # input is PSD array and frequency array

    # calculate the ERS and FDS
    load_spectrum.get_ers()
    load_spectrum.get_fds(k=10, C=1e80, p=6.3 * 1e10)
    
    # plot the results
    load_spectrum.plot_ers()
    load_spectrum.plot_fds()

    # or access the results directly
    ers = load_spectrum.ers
    fds = load_spectrum.fds
    f = load_spectrum.f0_range  # frequency vector
    

Random signals (time history)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of determining the ERS and FDS of a random signal, defined in the time domain. For time domain, two methods are available:
    - Convolution (directly from time history, using rainflow counting)
    - PSD averaging (converting time history to PSD and then to ERS and FDS)

Import random time history data:

.. code-block:: python

    import numpy as np

    _time_data = np.load('test_data/test_time_history.npy', allow_pickle=True)
    time_history_data = _time_data[:,1]
    t = _time_data[:,0] 
    dt = t[2] - t[1]

Use the package:

.. code-block:: python

    import FatigueDS
    
    # instantiate the Spectrum classes
    load_spectrum_1 = FatigueDS.Spectrum(freq_data=(20, 200, 5))  # convolution
    load_spectrum_2 = FatigueDS.Spectrum(freq_data=(20, 200, 5))  # psd averaging

    # set the random loads (input is time history array and time step)
    load_spectrum_1.set_random_load((time_history_data, dt), unit='g', method='convolution')
    load_spectrum_2.set_random_load((time_history_data, dt), unit='g', method='psd_averaging', bins=10)

    # calculate the ERS and FDS
    load_spectrum_1.get_ers()
    load_spectrum_1.get_fds(k=10, C=1e80, p=6.3 * 1e10)

    load_spectrum_2.get_ers()
    load_spectrum_2.get_fds(k=10, C=1e80, p=6.3 * 1e10)

    # plot the results

    load_spectrum_1.plot_ers(label='Time history (convolution)')
    load_spectrum_2.plot_ers(new_figure=False, label='Time history (PSD averaging)')
    
    load_spectrum_1.plot_fds(label='Time history (convolution)')
    load_spectrum_2.plot_fds(new_figure=False, label='Time history (PSD averaging)')

    # or access the results directly

    ers_1 = load_spectrum_1.ers
    fds_1 = load_spectrum_1.fds
    f_1 = load_spectrum_1.f0_range  # frequency vector

    ers_2 = load_spectrum_2.ers
    fds_2 = load_spectrum_2.fds
    f_2 = load_spectrum_2.f0_range  # frequency vector

Sine and sine-sweep signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of determining the ERS and FDS of a sine and sine-sweep signal:

.. code-block:: python

    import numpy as np
    import FatigueDS
    import matplotlib.pyplot as plt

    # instantiate classes
    load_spectrum_sine = FatigueDS.Spectrum(freq_data=(0, 2000, 5), damp=0.1)  # sine
    load_spectrum_sine_sweep = FatigueDS.Spectrum(freq_data=(0, 2000, 5), damp=0.1)  # sine sweep

    # set the sine and sine-sweep loads
    load_spectrum_sine.set_sine_load(sine_freq=500, amp=10, t_total=3600)  # t_total is is required only for FDS calculation.
    load_spectrum_sine_sweep.set_sine_sweep_load(const_amp=[5, 10, 20], const_f_range=[20, 100, 500, 1000], exc_type='acc', sweep_type='log', sweep_rate=1)

    # calculate the ERS and FDS
    load_spectrum_sine.get_ers()
    load_spectrum_sine_sweep.get_ers()

    load_spectrum_sine.get_fds(k=10, C=1e80, p=6.3 * 1e10)
    load_spectrum_sine_sweep.get_fds(k=10, C=1e80, p=6.3 * 1e10)

    # plot the results
    load_spectrum_sine.plot_ers(label='sine')
    load_spectrum_sine.plot_fds(label='sine')
    
    load_spectrum_sine_sweep.plot_ers(label='sine sweep')
    load_spectrum_sine_sweep.plot_fds(label='sine sweep')


References:
    1. C. Lalanne, Mechanical Vibration and Shock: Specification development,
    London, England: ISTE Ltd and John Wiley & Sons, 2009
