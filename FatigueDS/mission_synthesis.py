"""Mission synthesis: combine the ERS and FDS of the events of a life cycle profile
and derive a random vibration test PSD of reduced duration.

The method follows [Lalanne5]_ ch. 11 (sections 11.4.7 to 11.4.10). Page numbers
are the printed page numbers of the volume.

References
----------
.. [Lalanne4] C. Lalanne, *Mechanical Vibration and Shock Analysis, Vol. 4:
   Fatigue Damage*, 2nd ed., ISTE/Wiley, 2009.
.. [Lalanne5] C. Lalanne, *Mechanical Vibration and Shock Analysis, Vol. 5:
   Specification Development*, 2nd ed., ISTE/Wiley, 2009.
"""

import warnings

import numpy as np

from .spectrum import Spectrum


class MissionSynthesis:
    """Derive a random vibration test PSD from the events of a life cycle profile.

    Workflow: give the events (one :class:`~FatigueDS.spectrum.Spectrum` each, with
    ``ers`` and ``fds`` computed), then call :meth:`invert` and :meth:`check_ers`.

    - :meth:`combine`: reference FDS = sum of the event FDSs, reference ERS =
      envelope of the event ERSs (situations in series, [Lalanne5]_ p.306-307).
    - :meth:`invert`: test PSD of duration ``T_test`` with the reference FDS, by
      the iteration of [Lalanne5]_ p.313, eq. [11.11].
    - :meth:`check_ers`: test ERS against the reference ERS and, optionally, the
      SRS ([Lalanne5]_ p.314-316, Figs. 11.19-11.22).

    The FDS is proportional to the duration ([Lalanne4]_ p.142, eq. [4.41]), so for a
    random PSD event set ``T`` to the total exposure time of the event in the life
    cycle profile ([Lalanne5]_ p.325 computes the FDS of a measured PSD for the real
    duration of the stationary phase). The ERS of the event uses the same ``T``.
    """
    # ponytail: no repeats factor; set T to the total exposure (PSD events) or add a
    # time-history event n times. Add a repeats argument (FDS only) if asked.
    # ponytail: series situations only (p.307); add a parallel group (np.max of the
    # FDSs, p.306) if asked. No uncertainty or test factors (section 11.4.6, 11.5.1).

    def __init__(self, events=None):
        """
        :param events: list of :class:`~FatigueDS.spectrum.Spectrum` objects with
            ``ers`` and ``fds`` computed (default: no events)
        """
        self.events = list(events or [])

    def add_event(self, spectrum):
        """
        Add an event to the life cycle profile.

        :param spectrum: :class:`~FatigueDS.spectrum.Spectrum` with ``ers`` and ``fds`` computed

        :return: self
        """
        self.events.append(spectrum)
        return self

    def combine(self):
        """
        Compute the reference FDS and ERS of the life cycle profile.

        ``fds_ref`` is the sum of the event FDSs and ``ers_ref`` the envelope of the
        event ERSs ([Lalanne5]_ p.306, section 11.4.7 and p.307, section 11.4.8.2).
        All events must share ``f0_range``, damping, ``k``, ``C``, ``p`` and unit.
        ``f0_range`` must start above 0 Hz and be uniformly spaced, because the test
        PSD is defined on ``f0_range`` and ``tools.rms_sum`` uses one constant
        frequency step.

        :return: self
        """
        if not self.events:
            raise ValueError('no events: add at least one ``Spectrum`` event')
        f0 = np.array(self.events[0].f0_range, dtype=float)
        if any(e.f0_range.shape != f0.shape or not np.allclose(e.f0_range, f0) for e in self.events):
            raise ValueError('All events must have the same ``f0_range``')
        # Spectrum replaces f0 = 0 with 1e-3 Hz (tools.get_freq_range)
        if f0[0] <= 1e-3:
            raise ValueError('``f0_range`` must start above 0 Hz')
        # ponytail: uniform f0 only; for points per octave (p.313) evaluate the test
        # PSD on an internal uniform grid, or give rms_sum per-segment widths.
        if not np.allclose(np.diff(f0), f0[1] - f0[0]):
            raise ValueError('``f0_range`` must be uniformly spaced')
        params = {(e.k, e.C, e.p, e.damp, e.unit_scale) for e in self.events}
        if len(params) > 1:
            raise ValueError('All events must have the same k, C, p, damping and unit')

        self.f0_range = f0
        self.k, self.C, self.p, self.damp, self.unit_scale = params.pop()
        self.fds_ref = np.sum([e.fds for e in self.events], axis=0)
        self.ers_ref = np.max([e.ers for e in self.events], axis=0)
        if not np.all(np.isfinite(self.fds_ref) & (self.fds_ref > 0)):
            raise ValueError('The reference FDS must be finite and positive at every ``f0``')
        return self

    def _spectrum(self, psd, T):
        """Spectrum of a PSD defined on ``f0_range``, in the unit of the events."""
        s = Spectrum(freq_data=self.f0_range.copy(), damp=self.damp)
        s.set_random_load((psd, self.f0_range.copy()), T=T, unit='g' if self.unit_scale == 9.81 else 'ms2')
        return s

    def invert(self, T_test, max_iter=200, tol=0.05):
        """
        Compute the test PSD of duration ``T_test`` whose FDS equals the reference FDS.

        Iteration of [Lalanne5]_ p.313, section 11.4.9.2: the PSD is defined at the
        ``f0`` points of the reference FDS, starts at 0.1 (in the unit of the events)
        squared per Hz, and is updated with eq. [11.11]:

        ``PSD_1 = PSD_0 * (FDS_0 / FDS_1)**(2/b)``,

        with ``b`` the Basquin exponent (``k`` in FatigueDS notation), ``FDS_0`` the
        reference FDS and ``FDS_1`` the FDS of ``PSD_0``. The FDS is homogeneous of
        degree ``k/2`` in the PSD, so the start level has no effect after the first
        update. The loop stops when ``max|fds_test/fds_ref - 1| < tol`` or after
        ``max_iter`` updates, and the FDS of the last PSD is always computed.

        A sum of FDSs is not always the FDS of one PSD, and the convergence is slow
        where the PSD is small. The largest error is then found at the ends of
        ``f0_range``: compare ``fds_test`` with ``fds_ref`` ([Lalanne5]_ p.314,
        Fig. 11.18).

        The test PSD is defined on ``f0_range`` only. If an event has PSD content
        outside ``f0_range``, the iteration puts its damage on the end points:
        ``test_psd`` at the ends of ``f0_range`` becomes many times the event level,
        and the iteration may not converge. Make ``f0_range`` cover the frequency band
        of the events.

        :param T_test: test duration [s]
        :param max_iter: maximum number of updates (default: 200)
        :param tol: tolerance on the maximum relative FDS error (default: 0.05)

        :return: self. Sets ``T_test``, ``test_psd`` (on ``f0_range``, unit of the
            events squared per Hz), ``fds_test``, ``invert_error`` (maximum relative
            FDS error) and ``invert_n_iter`` (number of updates).
        """
        self.combine()
        psd = np.full_like(self.f0_range, 0.1)
        for n in range(max_iter + 1):
            s = self._spectrum(psd, T_test)
            s.get_fds(k=self.k, C=self.C, p=self.p)
            err = np.abs(s.fds / self.fds_ref - 1)
            if err.max() < tol or n == max_iter:
                break
            psd = psd * (self.fds_ref / s.fds)**(2 / self.k)  # [Lalanne5] p.313 eq [11.11]

        # ponytail: last iterate, no best-so-far tracking or stall stop; add if divergence is seen.
        self.T_test = T_test
        self.test_psd = psd
        self.fds_test = s.fds
        self.invert_error = err.max()
        self.invert_n_iter = n
        if self.invert_error >= tol:
            warnings.warn(
                f'FDS of the test PSD differs from the reference FDS by {self.invert_error:.1%} '
                f'at f0 = {self.f0_range[np.argmax(err)]:g} Hz after {n} updates (tol = {tol:g}). '
                'The error is usually largest at the ends of ``f0_range``, where the PSD is small. '
                'Compare ``fds_test`` with ``fds_ref`` ([Lalanne5] p.314, Fig. 11.18).', stacklevel=2)
        return self

    def check_ers(self, srs=None, tol=0.05):
        """
        Compare the ERS of the test PSD with the reference ERS and, optionally, the SRS.

        [Lalanne5]_ p.314-316, Figs. 11.19-11.22. A test ERS above the reference ERS is
        the expected result of a time reduction (Fig. 11.19) and gives no warning.
        A warning is given when:

        - the test ERS is below the reference ERS by more than ``tol``: the test
          duration is too long, reduce ``T_test`` (Fig. 11.22);
        - the test ERS is above the SRS: increase ``T_test``, or precede the test
          with a shock that envelopes the test ERS (Figs. 11.20-11.21).

        :param srs: shock response spectrum of the life cycle profile on ``f0_range``,
            with the damping and unit of the events (optional)
        :param tol: tolerance on the ERS ratio below 1 (default: 0.05)

        :return: self. Sets ``ers_test`` (unit of the events) and
            ``ers_ratio = ers_test / ers_ref``.
        """
        s = self._spectrum(self.test_psd, self.T_test)
        s.get_ers()
        self.ers_test = s.ers
        self.ers_ratio = self.ers_test / self.ers_ref

        i = np.nanargmin(self.ers_ratio)
        if self.ers_ratio[i] < 1 - tol:
            warnings.warn(
                f'Test ERS is below the reference ERS (ratio {self.ers_ratio[i]:.3f} at '
                f'f0 = {self.f0_range[i]:g} Hz): test duration too long, reduce ``T_test`` '
                '([Lalanne5] p.315-316, Fig. 11.22).', stacklevel=2)
        if srs is not None:
            r = self.ers_test / np.asarray(srs)
            j = np.nanargmax(r)
            if r[j] > 1:
                warnings.warn(
                    f'Test ERS exceeds the SRS (ratio {r[j]:.3f} at f0 = {self.f0_range[j]:g} Hz): '
                    'increase ``T_test`` or precede the test with a shock enveloping the test ERS '
                    '([Lalanne5] p.314-316, Figs. 11.20-11.21).', stacklevel=2)
        return self
