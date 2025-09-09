Getting started
===============

To install FatigueDS, use ``pip``:

.. code-block:: console

    $ pip install FatigueDS


Importing the package
----------------------

.. code-block:: python
    
    import FatigueDS


Creating a Spectrum object
-------------------------------------------

To calculate the Extreme Response Spectrum (ERS) and Fatigue Damage Spectrum (FDS), a Spectrum object must be created. The object is created by providing the frequency range and damping ratio.
Frequency range ``freq_data`` is defined by a tuple (``f0_start``, ``f0_stop``, ``f0_step``) in Hz and sets the frequency points where the ERS and FDS will be calculated. Alternatively, a frequency vector (array) can be passed as input. 
Damping ratio ``damp`` is a float value between 0 and 1.

.. code-block:: python

    load_spectrum = FatigueDS.Spectrum(freq_data=(f0_start, f0_stop, f0_step), damp)

    # or

    load_spectrum = FatigueDS.Spectrum(freq_vector, damp)


Setting the load signal
------------------------

Load signal is defined with different methods, depending on the type of signal.

Random signal (PSD)
~~~~~~~~~~~~~~~~~~~~

For the random signal defined by the Power Spectral Density, the following parameters must be provided:

* ``signal_data`` : A tuple (PSD array, frequency array),

* ``unit`` : unit of the PSD (``ms2`` or ``g``)

* ``T`` : time duration in seconds

.. code-block:: python

    load_spectrum.set_random_load((PSD, freq), unit, T)


Random signal (time history)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the random signal defined by the time history, the following parameters must be provided:

* ``signal_data`` : A tuple (time history array, time step)

* ``unit`` : unit of the time history (``ms2`` or ``g``)

* ``method`` : method of calculation.

Available methods are:

* ``convolution`` : Directly from time history

* ``psd_averaging`` : Conversion to PSD, then to ERS and FDS from PSD

.. code-block:: python

    load_spectrum.set_random_load((time_history, dt), unit, method)


Sine signal
~~~~~~~~~~~~

For the sine signal; frequency, amplitude, total time of the signal and an excitation type must be provided.

.. code-block:: python
    
    load_spectrum.set_sine_load(sine_freq, amp, t_total, exc_type)


Sine-sweep signal
~~~~~~~~~~~~~~~~~~

For the sine-sweep signal; amplitude, frequency range, excitation type, time step, sweep type and sweep rate must be provided.

.. code-block:: python

    load_spectrum.set_sine_sweep_load(const_amp, const_f_range, exc_type, dt, sweep_type, sweep_rate)




Calculating the ERS and FDS
----------------------------

After the load signal is set, the ERS and FDS can be calculated.

ERS is calculated by:

.. code-block:: python

    load_spectrum.get_ers()


FDS calculation requires additional material fatigue parameters: ``k``, ``C`` and ``p``. It is calculated by:

.. code-block:: python

    load_spectrum.get_fds(k, C, p)

The results are stored in the ``ers`` and ``fds`` attributes of the Spectrum object.

Accessing the results:

.. code-block:: python

    load_spectrum.ers

    load_spectrum.fds

    load_spectrum.f0_range  # frequency array

Plotting the results
-------------------------------

The results can be plotted by:

.. code-block:: python

    load_spectrum.plot_ers()

    load_spectrum.plot_fds()