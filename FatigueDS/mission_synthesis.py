import numpy as np
from scipy.special import gamma
import warnings
from .spectrum import Spectrum


def combine_fds(fds_list, repeats=None):
    """Combine event FDS by damage summation: sum_i repeats_i * fds_i.

    :param fds_list: list of 1-D FDS arrays, all the same length
    :param repeats: optional per-event occurrence multipliers (default all 1)
    :return: combined FDS array
    """
    if len(fds_list) == 0:
        raise ValueError("fds_list must contain at least one FDS array")
    arrs = [np.asarray(f, dtype=float) for f in fds_list]
    n = arrs[0].shape[0]
    if any(a.shape != (n,) for a in arrs):
        raise ValueError("all FDS arrays must be 1-D and the same length")
    if repeats is None:
        repeats = np.ones(len(arrs))
    repeats = np.asarray(repeats, dtype=float)
    if repeats.shape != (len(arrs),):
        raise ValueError("repeats length must match the number of FDS arrays")
    if np.any(repeats <= 0):
        raise ValueError("repeats must be positive")
    total = np.zeros(n)
    for a, r in zip(arrs, repeats):
        total += r * a
    return total


def envelope_ers(ers_list):
    """Combine event ERS by elementwise maximum (extreme-response envelope).

    :param ers_list: list of 1-D ERS arrays, all the same length
    :return: enveloped ERS array
    """
    if len(ers_list) == 0:
        raise ValueError("ers_list must contain at least one ERS array")
    arrs = [np.asarray(e, dtype=float) for e in ers_list]
    n = arrs[0].shape[0]
    if any(a.shape != (n,) for a in arrs):
        raise ValueError("all ERS arrays must be 1-D and the same length")
    return np.max(np.vstack(arrs), axis=0)


def invert_fds_to_psd(fds_ref, f0_range, k, C, p, Q, T_test):
    """Invert a reference FDS to an equivalent test acceleration PSD.

    Closed-form inverse of Lalanne (Specification Development, Vol.5) eq [4.9]:

        G(f0) = (2*w0**3 / Q) * [ C*fds_ref / (p**k * f0 * T_test * gamma(1+k/2)) ]**(2/k)

    with w0 = 2*pi*f0 and the narrow-band assumption n0+ = f0. Returns 0 where the
    reference damage is 0 or f0 is 0.

    :param fds_ref: reference fatigue damage spectrum (per f0)
    :param f0_range: natural-frequency axis [Hz]
    :param k: S-N slope (Basquin N*s**k = C)
    :param C: S-N constant
    :param p: stress/relative-displacement proportionality constant
    :param Q: quality factor
    :param T_test: test duration [s]
    :return: acceleration PSD G(f0) on f0_range [(m/s^2)^2/Hz]
    """
    fds_ref = np.asarray(fds_ref, dtype=float)
    f0 = np.asarray(f0_range, dtype=float)
    if T_test <= 0:
        raise ValueError("T_test must be positive")
    w0 = 2 * np.pi * f0
    psd = np.zeros_like(f0)
    mask = (fds_ref > 0) & (f0 > 0)
    inner = C * fds_ref[mask] / (p**k * f0[mask] * T_test * gamma(1 + k / 2))
    psd[mask] = (2 * w0[mask]**3 / Q) * inner**(2 / k)
    return psd


class MissionSynthesis:
    """Combine the FDS/ERS of several Spectrum events into reference curves and
    invert the reference FDS into an equivalent accelerated test PSD.

    All events must share the same (uniformly spaced) f0_range.
    """

    def __init__(self, events=None):
        """:param events: optional list of Spectrum (each with .fds and .ers computed)"""
        self.events = []
        if events is not None:
            for ev in events:
                self.add_event(ev)

    def add_event(self, spectrum, repeats=1):
        """Append a field event.

        :param spectrum: a Spectrum instance with .fds and .ers computed
        :param repeats: occurrence multiplier in the life profile (> 0)
        """
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.events.append((spectrum, float(repeats)))
        return self

    def combine(self):
        """Build reference curves: fds_ref = sum(repeats*fds), ers_ref = max(ers)."""
        if not self.events:
            raise ValueError("no events added")
        specs = [s for s, _ in self.events]
        reps = [r for _, r in self.events]
        f0 = np.asarray(specs[0].f0_range, dtype=float)
        for s in specs[1:]:
            if not np.allclose(np.asarray(s.f0_range, dtype=float), f0):
                raise ValueError("all events must share the same f0_range")
        for s in specs:
            if not hasattr(s, 'fds'):
                raise ValueError("each event must have .fds (run get_fds first)")
            if not hasattr(s, 'ers'):
                raise ValueError("each event must have .ers (run get_ers first)")
        self.f0_range = f0
        self.fds_ref = combine_fds([s.fds for s in specs], repeats=reps)
        self.ers_ref = envelope_ers([s.ers for s in specs])
        return self

    def _resolve_param(self, name, override):
        if override is not None:
            return override
        vals = [getattr(s, name) for s, _ in self.events if hasattr(s, name)]
        if not vals:
            raise ValueError(f"parameter '{name}' not found on events; pass it explicitly")
        first = vals[0]
        if any(not np.isclose(v, first) for v in vals):
            raise ValueError(f"events have inconsistent '{name}'; pass it explicitly to invert()")
        return first

    def invert(self, T_test, k=None, C=None, p=None, Q=None):
        """Invert the reference FDS to a test acceleration PSD for duration T_test.

        Material params default to the (consistent) values on the events; pass any of
        k, C, p, Q to override. Sets self.test_psd and self.test_psd_freq.
        """
        if not hasattr(self, 'fds_ref'):
            raise ValueError("call combine() before invert()")
        if T_test <= 0:
            raise ValueError("T_test must be positive")
        self.k = self._resolve_param('k', k)
        self.C = self._resolve_param('C', C)
        self.p = self._resolve_param('p', p)
        self.Q = self._resolve_param('Q', Q)
        self.T_test = T_test
        self.test_psd = invert_fds_to_psd(
            self.fds_ref, self.f0_range, k=self.k, C=self.C, p=self.p, Q=self.Q, T_test=T_test)
        self.test_psd_freq = self.f0_range
        return self

    def check_ers(self, tol=0.05):
        """Compare the derived test ERS to the reference ERS (over-test guard).

        Sets self.ers_test and self.ers_ratio = ers_test / ers_ref (0 where ers_ref==0).
        Warns where ers_ratio > 1 + tol (the accelerated test exceeds the field extreme
        response and may excite field-absent failure modes).
        """
        if not hasattr(self, 'test_psd'):
            raise ValueError("call invert() before check_ers()")
        s = Spectrum(freq_data=self.f0_range, Q=self.Q)
        s.set_random_load((self.test_psd, self.test_psd_freq), unit='ms2', T=self.T_test)
        s.get_ers()
        self.ers_test = s.ers
        ratio = np.where(self.ers_ref > 0, self.ers_test / self.ers_ref, 0.0)
        self.ers_ratio = ratio
        if np.any(ratio > 1 + tol):
            i = int(np.argmax(ratio))
            warnings.warn(
                f"test ERS exceeds reference ERS (max ratio {ratio[i]:.2f} at "
                f"{self.f0_range[i]:.1f} Hz): possible over-test")
        return self
