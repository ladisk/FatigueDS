Mission synthesis (test tailoring)
==================================

This example shows how to combine the fatigue damage of several real-life
vibration events into a reference Fatigue Damage Spectrum (FDS) and Extreme
Response Spectrum (ERS), and then **invert** the reference FDS into an
equivalent, accelerated laboratory test PSD.

The workflow follows C. Lalanne, *Mechanical Vibration and Shock: Specification
Development* (ISTE / Wiley, 2009):

* the FDS of the individual events are **summed** (fatigue damage accumulates),
* the ERS of the events are **enveloped** (extreme stress is the worst single event),
* the reference FDS is **inverted** to a test PSD for a chosen (usually shorter)
  test duration, and
* the resulting test ERS is **compared** to the reference ERS to guard against
  over-testing.

.. note::
   The "road transport + engine" life profile below is an *illustrative* scenario
   (the numbers are chosen to demonstrate the workflow). It is in the spirit of
   Lalanne's own life-profile examples, which combine named real-world environments —
   e.g. truck transport + missile flight (Example 11.2), aircraft + helicopter
   (Example 11.3), and a full truck (good road / bad road / railroad-crossing) +
   aircraft + helicopter profile (Example 11.4).

Import the required packages
----------------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import FatigueDS

Define the life profile as a set of events
------------------------------------------

Each event is an ordinary :class:`~FatigueDS.Spectrum`. Build one per event, with
its own load and duration, and compute its ERS and FDS. All events must share the
same natural-frequency range (``freq_data``) and the same material parameters.

.. code-block:: python

    freq_range = (20, 480, 5)        # shared natural-frequency axis [Hz]
    Q = 10                           # quality factor
    k, C, p = 7, 1.0, 1.0            # S-N slope, S-N constant, stress/displacement constant

    # Event 1: road transport - broadband 20-500 Hz, 3 hours
    freq1 = np.arange(20, 500, 1.0)
    psd1 = np.full_like(freq1, 0.5)  # (m/s^2)^2/Hz
    road = FatigueDS.Spectrum(freq_data=freq_range, Q=Q)
    road.set_random_load((psd1, freq1), unit='ms2', T=3 * 3600)
    road.get_ers()
    road.get_fds(k=k, C=C, p=p)

    # Event 2: engine running - a resonance bump around 275 Hz, 1 hour, occurs 5 times
    freq2 = np.arange(20, 500, 1.0)
    psd2 = 2.0 * np.exp(-0.5 * ((freq2 - 275) / 70) ** 2)   # smooth resonance bump
    engine = FatigueDS.Spectrum(freq_data=freq_range, Q=Q)
    engine.set_random_load((psd2, freq2), unit='ms2', T=1 * 3600)
    engine.get_ers()
    engine.get_fds(k=k, C=C, p=p)

.. note::
    The FDS theory uses SI base units, so define the events with ``unit='ms2'``
    (acceleration PSD in ``(m/s^2)^2/Hz``).

Combine the events
------------------

Add the events to a :class:`~FatigueDS.MissionSynthesis` object and call
``combine()``. The ``repeats`` argument is the number of occurrences of an event
in the life profile. ``combine()`` sums the FDS and envelopes the ERS.

.. code-block:: python

    ms = FatigueDS.MissionSynthesis()
    ms.add_event(road, repeats=1)
    ms.add_event(engine, repeats=5)   # the engine event occurs 5 times
    ms.combine()

    # reference curves are available as ms.fds_ref, ms.ers_ref, ms.f0_range
    ms.plot_fds()
    road.plot_fds(new_figure=False)
    engine.plot_fds(new_figure=False)
    plt.legend(['reference (combined)', 'road (1x)', 'engine (5x)'])

Invert the reference FDS to a test PSD
--------------------------------------

``invert(T_test)`` derives the acceleration PSD that reproduces the reference
fatigue damage over the chosen test duration ``T_test``. A shorter test duration
gives a higher level (an accelerated test). The material parameters default to the
values stored on the events; pass ``k``, ``C``, ``p`` or ``Q`` to override.

By default it uses Lalanne's **iteration method** (Vol. 5 eq [11.11]): it repeatedly
adjusts the PSD until the *full forward* FDS of the candidate PSD matches the
reference (so it accounts for each oscillator's off-resonance response). The achieved
fit is reported in ``ms.invert_error`` (max relative FDS error) and ``ms.invert_n_iter``.
A fast diagonal approximation is available with ``method='closed_form'`` (the inverse
of eq [4.9]); it ignores off-resonance coupling and is mainly useful as a quick
estimate.

.. code-block:: python

    T_test = 30 * 60                 # 30-minute accelerated test [s]
    ms.invert(T_test=T_test)         # method='iteration' by default

    # derived test PSD is available as ms.test_psd on ms.test_psd_freq
    print('FDS reproduced to', f'{ms.invert_error:.1%}', 'in', ms.invert_n_iter, 'iterations')
    ms.plot_test_psd()

.. note::
   A reference FDS obtained by *summing* several events is generally not exactly the
   FDS of any single stationary PSD (the FDS is non-linear in the PSD for ``k != 2``),
   so a small residual at sharp FDS features is expected and irreducible.

Over-test guard
---------------

Accelerating a test raises its instantaneous levels. ``check_ers()`` recomputes the
ERS of the derived test PSD and compares it to the reference ERS, warning where the
test would exceed the field extreme response (which could excite failure modes not
present in service).

.. code-block:: python

    ms.check_ers()
    print('max test/reference ERS ratio:', np.max(ms.ers_ratio))

For the aggressive ~16x time compression above (about 8 hours of life into a 30
minute test, with an S-N slope ``k = 7``), the test ERS exceeds the reference by
roughly 1.4x and ``check_ers()`` emits a warning. Lengthening ``T_test`` (a milder
acceleration) reduces the over-test.

.. note::
   Use smooth field PSDs (as measured). A PSD with a hard band edge makes the
   reference FDS jump almost discontinuously there; because each oscillator responds
   over a finite bandwidth (~f0/Q), the inversion cannot reproduce such a step and
   produces a non-physical notch in the test PSD just below the edge. In practice the
   raw inverted PSD is then **enveloped** into a few breakpoints to form the final
   specification.
