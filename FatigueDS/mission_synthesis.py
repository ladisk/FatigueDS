import numpy as np
from scipy.special import gamma
import warnings
import matplotlib.pyplot as plt
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


def invert_fds_to_psd_iterative(fds_ref, f0_range, k, C, p, Q, T_test,
                                max_iter=200, patience=15):
    """Invert a reference FDS to a test PSD by Lalanne's iteration method.

    Lalanne (Specification Development, Vol.5) eq [11.11]: starting from an initial
    PSD, repeatedly rescale each level by the ratio of the target to the achieved FDS,

        G_{n+1}(f_i) = G_n(f_i) * ( FDS_ref(f_i) / FDS(G_n)(f_i) )**(2/k),

    where FDS(G_n) is the *full* forward FDS of the candidate PSD (computed with
    ``Spectrum``, so it includes the off-resonance response of each oscillator). This
    reproduces the reference FDS far better than the diagonal closed form
    ``invert_fds_to_psd`` (used here as the initial guess), which ignores off-resonance
    coupling. Lalanne recommends this iteration over the matrix-inversion method
    (eq [11.2]), which is prone to numerical instability.

    The best-fitting PSD found is returned. Note that a reference FDS obtained by
    *summing* events (mission synthesis) is generally not exactly the FDS of any single
    stationary PSD (FDS is non-linear in the PSD for k != 2), so a small residual at
    sharp FDS features is expected and irreducible; the iteration is stopped once it
    stops improving.

    :param fds_ref: reference fatigue damage spectrum (per f0)
    :param f0_range: natural-frequency axis [Hz] (uniformly spaced)
    :param k, C, p, Q: material/structure parameters (see ``invert_fds_to_psd``)
    :param T_test: test duration [s]
    :param max_iter: maximum number of iterations
    :param patience: stop after this many iterations without improvement
    :return: (psd, n_iter, error) - best PSD on f0_range, iterations used, and the
        achieved error = max|FDS(psd)/FDS_ref - 1| over the non-zero reference
    """
    fds_ref = np.asarray(fds_ref, dtype=float)
    f0 = np.asarray(f0_range, dtype=float)
    if T_test <= 0:
        raise ValueError("T_test must be positive")
    m = fds_ref > 0
    G = invert_fds_to_psd(fds_ref, f0, k=k, C=C, p=p, Q=Q, T_test=T_test)  # warm start
    best_G, best_err, stale, n_iter = G.copy(), np.inf, 0, 0
    for it in range(1, max_iter + 1):
        s = Spectrum(freq_data=f0, Q=Q)
        s.set_random_load((G, f0), unit='ms2', T=T_test)
        s.get_fds(k=k, C=C, p=p)
        cand = s.fds
        n_iter = it
        err = np.max(np.abs(cand[m] / fds_ref[m] - 1.0)) if np.any(m) else 0.0
        if err < best_err * (1 - 1e-3):
            best_err, best_G, stale = err, G.copy(), 0
        else:
            stale += 1
            if stale >= patience:
                break
        ratio = np.ones_like(G)
        good = m & (cand > 0)
        ratio[good] = fds_ref[good] / cand[good]
        G = G * ratio ** (2.0 / k)
    return best_G, n_iter, best_err


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

    def invert(self, T_test, k=None, C=None, p=None, Q=None,
               method='iteration', max_iter=200, warn_tol=0.25):
        """Invert the reference FDS to a test acceleration PSD for duration T_test.

        Material params default to the (consistent) values on the events; pass any of
        k, C, p, Q to override. Sets self.test_psd and self.test_psd_freq.

        :param method: 'iteration' (default) - Lalanne's iterative method (eq [11.11]),
            whose result reproduces the reference FDS through the full forward response
            (off-resonance included); 'closed_form' - the fast diagonal/narrow-band
            approximation (inverse of eq [4.9]), which ignores off-resonance coupling.
        :param max_iter: maximum iterations for the 'iteration' method
        :param warn_tol: warn if the iteration's achieved FDS error exceeds this (the
            reference may not be well representable by a single stationary PSD)

        For 'iteration', also sets self.invert_n_iter and self.invert_error (the
        achieved max|FDS(test_psd)/FDS_ref - 1|).
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
        self.method = method
        if method == 'iteration':
            self.test_psd, self.invert_n_iter, self.invert_error = invert_fds_to_psd_iterative(
                self.fds_ref, self.f0_range, k=self.k, C=self.C, p=self.p, Q=self.Q,
                T_test=T_test, max_iter=max_iter)
            if self.invert_error > warn_tol:
                warnings.warn(
                    f"reference FDS reproduced only to {self.invert_error:.0%} at best; "
                    f"it may not be representable by a single stationary PSD "
                    f"(e.g. a sharply peaked or sine-derived reference FDS)")
        elif method == 'closed_form':
            self.test_psd = invert_fds_to_psd(
                self.fds_ref, self.f0_range, k=self.k, C=self.C, p=self.p, Q=self.Q, T_test=T_test)
        else:
            raise ValueError("method must be 'iteration' or 'closed_form'")
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

    def plot_fds(self, new_figure=True, **kwargs):
        """Plot the reference (combined) fatigue damage spectrum."""
        if not hasattr(self, 'fds_ref'):
            raise ValueError("call combine() before plot_fds()")
        if new_figure:
            plt.figure()
        plt.semilogy(self.f0_range, self.fds_ref, **kwargs)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('FDS [Damage]')
        plt.title('Reference Fatigue Damage Spectrum')
        plt.grid(visible=True)

    def plot_ers(self, new_figure=True, **kwargs):
        """Plot the reference (enveloped) extreme response spectrum."""
        if not hasattr(self, 'ers_ref'):
            raise ValueError("call combine() before plot_ers()")
        if new_figure:
            plt.figure()
        plt.plot(self.f0_range, self.ers_ref, **kwargs)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('ERS [m/s²]')
        plt.title('Reference Extreme Response Spectrum')
        plt.grid(visible=True)

    def plot_test_psd(self, new_figure=True, **kwargs):
        """Plot the derived equivalent test PSD."""
        if not hasattr(self, 'test_psd'):
            raise ValueError("call invert() before plot_test_psd()")
        if new_figure:
            plt.figure()
        plt.semilogy(self.test_psd_freq, self.test_psd, **kwargs)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [(m/s²)²/Hz]')
        plt.title('Equivalent Test PSD')
        plt.grid(visible=True)
