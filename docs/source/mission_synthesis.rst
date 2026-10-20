Mission synthesis
=================

Mission synthesis gives one random vibration test that has the same fatigue damage
as all the vibration events of a life cycle profile, in a shorter test duration.
The method is from C. Lalanne, *Mechanical Vibration and Shock Analysis*, 2nd ed.,
Vol. 5: *Specification Development*, ISTE/Wiley, 2009, ch. 11 (printed pages):

* The reference FDS is the sum of the event FDSs. The reference ERS is the envelope
  of the event ERSs (sections 11.4.7 and 11.4.8.2, p.306-307).
* The test PSD comes from the iteration of eq. [11.11] (section 11.4.9.2, p.313).
* The FDS and ERS of the test PSD are compared with the reference spectra
  (section 11.4.10, p.313-316).

The example is Example 11.2 of the book (p.316-319): a truck transport of 20 hours
and a missile flight of 5 minutes, with Q = 10 and b = 8. The digitized PSDs of the
book (Figs. 11.23 and 11.24) are in the ``test_data`` folder of the FatigueDS
repository: run the example from the root folder of the repository.

Define the events
-----------------

Each event is a :class:`~FatigueDS.Spectrum` with its ERS and FDS. All events must
have the same ``freq_data``, damping, material parameters and unit. ``freq_data``
must start above 0 Hz and be uniformly spaced. The test PSD is defined only at these
frequencies, so make them cover the frequency band of the events (here about 3 to 1035 Hz).
For a random PSD event, set ``T`` to the total time of the event in the life cycle
profile. The digitized PSDs have a lower rms value than the values printed in the
figures, so the example scales them to the printed values.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import FatigueDS

    events = []
    for name, rms, T in [('truck', 3.024746, 20 * 3600),     # Fig. 11.23, 20 h
                         ('missile', 27.59684, 5 * 60)]:     # Fig. 11.24, 5 min
        f, psd = np.loadtxt(f'test_data/lalanne_ex11_2_psd_{name}.csv', delimiter=',').T
        freq = np.arange(np.ceil(f[0]), f[-1], 1.0)  # uniform PSD frequency vector [Hz]
        psd = np.interp(freq, f, psd)
        psd *= rms**2 / np.trapezoid(psd, freq)      # [(m/s^2)^2/Hz], printed rms [m/s^2]

        event = FatigueDS.Spectrum(freq_data=(2, 1047, 5), Q=10)
        event.set_random_load((psd, freq), unit='ms2', T=T)
        event.get_ers()
        event.get_fds(k=8, C=1, p=1)
        events.append(event)
    truck, missile = events

Combine the events
------------------

.. code-block:: python

    ms = FatigueDS.MissionSynthesis([truck, missile]).combine()

    plt.semilogy(ms.f0_range, truck.fds, label='truck')
    plt.semilogy(ms.f0_range, missile.fds, label='missile')
    plt.semilogy(ms.f0_range, ms.fds_ref, 'k--', label='reference (sum)')
    plt.xlabel('f0 [Hz]')
    plt.ylabel('FDS [damage]')
    plt.legend()

The truck FDS is the larger one up to 22 Hz and the missile FDS above it (book
Fig. 11.27: crossing at about 24 Hz). The missile ERS is the envelope from 22 Hz.

Test PSD for 5 hours
--------------------

``invert(T_test)`` finds the PSD on ``f0_range`` whose FDS over ``T_test`` is equal
to the reference FDS, and its FDS ``fds_test``. Compare it with ``fds_ref`` (Fig. 11.18).

.. code-block:: python

    ms.invert(T_test=5 * 3600)
    print(f'{ms.invert_n_iter} updates, FDS error {ms.invert_error:.1%}')
    # 100 updates, FDS error 5.0%

    plt.figure()
    plt.semilogy(ms.f0_range, ms.fds_ref, label='reference')
    plt.semilogy(ms.f0_range, ms.fds_test, '--', label='test PSD, 5 h')
    plt.legend()

The iteration stops when the largest relative FDS error is less than ``tol``
(default 0.05), or gives a warning after ``max_iter`` updates. The largest error is
usually at the ends of ``f0_range``, where the PSD is small.

Check the ERS
-------------

``check_ers()`` calculates the ERS of the test PSD and the ratio
``ers_ratio = ers_test / ers_ref``. The book gives these cases (p.314-316,
Figs. 11.19-11.22). SRS is the shock response spectrum of the life cycle profile:

* SRS > test ERS > reference ERS: the ideal case. A shorter test gives a higher ERS.
* Test ERS > SRS: the exaggeration is too large. Increase ``T_test``, or start the
  test with a shock that envelopes the test ERS.
* Test ERS > reference ERS > SRS: do not reduce the duration too much.
* Reference ERS > test ERS: the test is too long. Reduce ``T_test``.

``check_ers()`` gives a warning if the ratio is less than ``1 - tol`` (default
``tol=0.05``). If you give ``srs`` (on ``f0_range``, with the damping and unit of the
events), it also gives a warning where the test ERS is more than the SRS.

.. code-block:: python

    ms.check_ers()
    # UserWarning: Test ERS is below the reference ERS (ratio 0.702 at f0 = 327 Hz):
    # test duration too long, reduce ``T_test`` ...

This is the result of the book (Fig. 11.26). The 5 hour test spreads the damage of
the 5 minute missile flight over 5 hours, so the levels and the ERS decrease. The peak
of the test ERS is 0.70 of the peak of the reference ERS (757 and 1076 m/s² at
172 Hz). In the book it is 0.71 (743 and 1052 m/s²).

Test PSD for 5 minutes
----------------------

.. code-block:: python

    ms.invert(T_test=5 * 60).check_ers()
    r = ms.ers_ratio[ms.f0_range > 25]
    print(f'ERS ratio above 25 Hz: {r.min():.2f} to {r.max():.2f}')
    # ERS ratio above 25 Hz: 1.00 to 1.02

    plt.figure()
    plt.plot(ms.f0_range, ms.ers_ref, label='reference (envelope)')
    plt.plot(ms.f0_range, ms.ers_test, '--', label='test PSD, 5 min')
    plt.xlabel('f0 [Hz]')
    plt.ylabel('ERS [m/s²]')
    plt.legend()

Above 25 Hz the two ERSs are very close (book Fig. 11.28). Below 25 Hz the test ERS
is up to 1.53 times the reference ERS (at 17 Hz, where the truck ERS is the envelope):
the test puts 20 hours of truck damage into 5 minutes.

Compared with the digitized book curves, the median ratio is 1.01 for the reference
ERS (Fig. 11.26), 0.96 for the 5 hour test ERS (Fig. 11.26) and 0.99 for the 5 minute
test ERS (Fig. 11.28).
