Random signal
================

Here is an example of determining the ERS and the FDS of a random signal, defined in both the time and frequency domains.

Import the required packages
----------------------------


.. code-block:: python

    import numpy as np
    import FatigueDS
    import matplotlib.pyplot as plt
    import pyExSi as es

Spectrum object
-------------------------------

The Spectrum object contains the data required for the calculation of the Extreme Response Spectrum (ERS) and the Fatigue Damage Spectrum (FDS). It enables calculations for random signals that are defined using either the PSD or time-history.
For time-history signals, two methods are available. ERS and FDS can be determined directly from time history using convolution, or by first converting the history into PSD and calculating spectra from the PSD.

This example demonstrates ERS and FDS calculations using all three available methods:

* from PSD

* from time-history using convolution (directly from time history)

* from time-history using PSD averaging (conversion to PSD, then ERS and FDS from PSD)

The ERS and FDS results from all methods are plotted for comparison.

Random signal (flat-shaped PSD)
--------------------------------

Generate data
~~~~~~~~~~~~~

Random signal is generated using PyExSi.

.. code-block:: python

    fs = 5000  # sampling frequency [Hz]
    time = 500

    N = int(time * fs)  # number of data points of time signal
    t = np.arange(0, N) / fs  # time vector

    # define frequency vector and one-sided flat-shaped PSD
    freq_flat = np.arange(0, fs / 2, 1 / time)  # frequency vector
    freq_lower = 200  # PSD lower frequency limit  [Hz]
    freq_upper = 1000  # PSD upper frequency limit [Hz]
    PSD_flat = es.get_psd(freq_flat, freq_lower, freq_upper, variance=800)  # one-sided flat-shaped PSD

    # get gaussian stationary signal
    gaussian_signal = es.random_gaussian(N, PSD_flat, fs)

    # for faster calculation, we can downsample the signal
    # PSD does not need high freq. resolution for good results
    PSD_flat = PSD_flat[::100]
    freq_flat = freq_flat[::100]

Plot the generated signal:

.. code-block:: python

    plt.plot(freq_flat, PSD_flat)
    plt.xlabel('f [Hz]')
    plt.ylabel('(m/s²)²/Hz')
    plt.show()

    plt.plot(t, gaussian_signal)
    plt.xlabel('t [s]')
    plt.ylabel('m/s²')
    plt.show()

Instantiate the Spectrum object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Object is instantiated with inputs:

* ``freq_data``: tuple containing (``f0_start``, ``f0_stop``, ``f0_step``) [Hz] or a frequency vector (array), defining the range where the ERS and FDS will be calculated.

*  damping ratio ``damp`` or damping Q-factor ``Q``.

Three objects are instantiated for comparison of all three methods:

.. code-block:: python
    
    load_spectrum_1 = FatigueDS.Spectrum(freq_data=(100, 1100, 20), damp=0.05)  # PSD
    load_spectrum_2 = FatigueDS.Spectrum(freq_data=(100, 1100, 20), damp=0.05)  # Time history (convolution)
    load_spectrum_3 = FatigueDS.Spectrum(freq_data=(100, 1100, 20), damp=0.05)  # Time history (psd averaging)

Set the random load
~~~~~~~~~~~~~~~~~~~

The random load is defined using the ``set_random_load`` method. Either a time history or a PSD must be provided as input. The class method automatically determines whether the input is a time history or a PSD based on its type:

* PSD: input is tuple containing (psd data (array), frequency vector (array)).

* Time history: input is tuple containing (time history data (array), dt (scalar)).

If time history is given as input, method of spectra calculation must also be defined. Available methods are:

* ``convolution`` (directly from time history)

* ``psd_averaging`` (conversion to PSD, then ERS and FDS from PSD)

.. code-block:: python

    load_spectrum_1.set_random_load((PSD_flat, freq_flat), unit='ms2', T=500)  # input is tuple (psd array, freq array)
    load_spectrum_2.set_random_load((gaussian_signal, 1 / fs), unit='ms2', method='convolution')  # input is tuple (psd data, frequency vector)
    load_spectrum_3.set_random_load((gaussian_signal, 1 / fs), unit='ms2', method='psd_averaging', bins=500)  # input is tuple (psd data, frequency vector)

Get the ERS and FDS
~~~~~~~~~~~~~~~~~~~~

ERS and FDS are calculated with the ``get_ers`` and ``get_fds`` methods. For the FDS calculation, the additional material fatigue parameters ``k``, ``C`` and ``p`` must be provided.

.. code-block:: python
    
    load_spectrum_1.get_ers()
    load_spectrum_2.get_ers()
    load_spectrum_3.get_ers()

    k = 10
    C = 1e80
    p = 6.3 * 1e10

    load_spectrum_1.get_fds(k=k, C=C, p=p)
    load_spectrum_2.get_fds(k=k, C=C, p=p)
    load_spectrum_3.get_fds(k=k, C=C, p=p)

The results are stored in the ``ers`` and ``fds`` attributes of the object, while the frequency vector is stored in the ``f0_range`` attribute.

.. code-block:: python

    load_spectrum_1.ers
    load_spectrum_1.fds
    load_spectrum_1.f0_range

Plot the results
~~~~~~~~~~~~~~~~

ERS and FDS are plotted for all three methods.

.. code-block:: python

    load_spectrum_1.plot_ers(label='PSD')
    load_spectrum_2.plot_ers(new_figure=False, label='Time history (convolution)')
    load_spectrum_3.plot_ers(new_figure=False, label='Time history (PSD averaging)')

    load_spectrum_1.plot_fds(label='PSD')
    load_spectrum_2.plot_fds(new_figure=False, label='Time history (convolution)')
    load_spectrum_3.plot_fds(new_figure=False, label='Time history (PSD averaging)')


