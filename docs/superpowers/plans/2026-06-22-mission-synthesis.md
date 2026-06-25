# Mission Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `MissionSynthesis` workflow that combines the FDS/ERS of multiple `Spectrum` events into reference curves and inverts the reference FDS into an equivalent accelerated test PSD, with an over-test guard against the reference ERS.

**Architecture:** One new module `FatigueDS/mission_synthesis.py` containing pure helper functions (`combine_fds`, `envelope_ers`, `invert_fds_to_psd`) and a thin stateful `MissionSynthesis` class that orchestrates them and reuses the existing `Spectrum` class for the ERS feedback. Exported from `FatigueDS/__init__.py`.

**Tech Stack:** Python ≥3.10, numpy, scipy (`scipy.special.gamma`), matplotlib — all already dependencies. No new dependencies.

## Global Constraints

- Package requires `numpy>=2.0`, `scipy>=1.0.0`, `matplotlib>=3.0.0`, Python `>=3.10`. No new dependencies.
- Notation maps Lalanne `b → k`, `K → p`, `C` unchanged (consistent with `Spectrum.get_fds`).
- FDS theory uses SI base units; PSD levels are acceleration `(m/s²)²/Hz`.
- Inversion is the closed-form inverse of Lalanne Vol.5 eq [4.9] (narrow-band, `n0+ = f0`).
- `f0_range` is assumed uniformly spaced (inherited from `Spectrum`'s PSD handling).
- Commit messages: conventional style; do NOT add any `Co-Authored-By` trailer.
- Tests run from the repo root: `python -m pytest tests/<file> -v`.

---

## File Structure

- `FatigueDS/mission_synthesis.py` — **new**. Pure helpers + `MissionSynthesis` class.
- `FatigueDS/__init__.py` — **modify**. Export `MissionSynthesis` and the helpers.
- `tests/test_mission_synthesis.py` — **new**. All tests for this feature.

---

### Task 1: Pure combination helpers (`combine_fds`, `envelope_ers`)

**Files:**
- Create: `FatigueDS/mission_synthesis.py`
- Test: `tests/test_mission_synthesis.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `combine_fds(fds_list: list[np.ndarray], repeats: list|np.ndarray|None = None) -> np.ndarray` — damage summation `Σ repeats_i · fds_i`.
  - `envelope_ers(ers_list: list[np.ndarray]) -> np.ndarray` — elementwise max.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mission_synthesis.py`:

```python
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

from FatigueDS.mission_synthesis import combine_fds, envelope_ers


def test_combine_fds_sums():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 0.5, 0.5])
    assert np.allclose(combine_fds([a, b]), np.array([1.5, 2.5, 3.5]))


def test_combine_fds_repeats():
    a = np.array([1.0, 2.0, 3.0])
    assert np.allclose(combine_fds([a], repeats=[2]), np.array([2.0, 4.0, 6.0]))


def test_combine_fds_length_mismatch():
    with pytest.raises(ValueError):
        combine_fds([np.array([1.0, 2.0]), np.array([1.0])])


def test_combine_fds_bad_repeats():
    with pytest.raises(ValueError):
        combine_fds([np.array([1.0, 2.0])], repeats=[0])


def test_envelope_ers_max():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([4.0, 2.0, 3.0])
    assert np.allclose(envelope_ers([a, b]), np.array([4.0, 5.0, 3.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission_synthesis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'FatigueDS.mission_synthesis'`

- [ ] **Step 3: Write minimal implementation**

Create `FatigueDS/mission_synthesis.py`:

```python
import numpy as np
from scipy.special import gamma


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mission_synthesis.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add FatigueDS/mission_synthesis.py tests/test_mission_synthesis.py
git commit -m "feat: add combine_fds and envelope_ers helpers"
```

---

### Task 2: FDS inversion helper (`invert_fds_to_psd`)

**Files:**
- Modify: `FatigueDS/mission_synthesis.py`
- Test: `tests/test_mission_synthesis.py`

**Interfaces:**
- Consumes: `scipy.special.gamma` (already imported in Task 1).
- Produces:
  - `invert_fds_to_psd(fds_ref: np.ndarray, f0_range: np.ndarray, k: float, C: float, p: float, Q: float, T_test: float) -> np.ndarray` — acceleration PSD `G(f0)` on `f0_range`, the closed-form inverse of Lalanne Vol.5 eq [4.9]:
    `G(f0) = (2*ω0³/Q) * [ C*fds_ref / (p**k * f0 * T_test * Γ(1+k/2)) ]**(2/k)`, with `ω0 = 2π f0`; `G = 0` where `fds_ref == 0` or `f0 == 0`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_synthesis.py`:

```python
import FatigueDS
from FatigueDS.mission_synthesis import invert_fds_to_psd


def test_invert_roundtrip_closure():
    """Forward FDS (Spectrum) -> invert -> recover the original flat PSD in-band.
    Wide band + high Q so the narrow-band inverse [4.9] matches the full integral."""
    df = 0.5
    freq = np.arange(10.0, 4000.0 + df, df)
    G = 2.0
    psd = np.full_like(freq, G)
    f0 = np.array([500.0, 1000.0, 1500.0])
    Q, k, C, p, T = 50, 1.0, 1.0, 1.0, 3600.0

    s = FatigueDS.Spectrum(freq_data=f0, Q=Q)
    s.set_random_load((psd, freq), unit='ms2', T=T)
    s.get_fds(k=k, C=C, p=p)

    recovered = invert_fds_to_psd(s.fds, f0, k=k, C=C, p=p, Q=Q, T_test=T)
    assert np.allclose(recovered, G, rtol=0.05)


def test_invert_zero_fds_gives_zero_psd():
    f0 = np.array([100.0, 200.0])
    fds = np.array([0.0, 0.0])
    out = invert_fds_to_psd(fds, f0, k=8, C=1.0, p=1.0, Q=10, T_test=3600.0)
    assert np.allclose(out, 0.0)


def test_invert_acceleration_scaling():
    """Shorter test duration -> higher PSD by factor (T/T_test)**(2/k)."""
    f0 = np.array([300.0, 600.0])
    fds = np.array([1e-20, 2e-20])
    k = 8
    g_full = invert_fds_to_psd(fds, f0, k=k, C=1.0, p=1.0, Q=10, T_test=3600.0)
    g_fast = invert_fds_to_psd(fds, f0, k=k, C=1.0, p=1.0, Q=10, T_test=360.0)
    assert np.allclose(g_fast / g_full, 10 ** (2 / k))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission_synthesis.py -k invert -v`
Expected: FAIL — `ImportError: cannot import name 'invert_fds_to_psd'`

- [ ] **Step 3: Write minimal implementation**

Append to `FatigueDS/mission_synthesis.py` (after `envelope_ers`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mission_synthesis.py -k invert -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add FatigueDS/mission_synthesis.py tests/test_mission_synthesis.py
git commit -m "feat: add invert_fds_to_psd (closed-form inverse of Lalanne eq [4.9])"
```

---

### Task 3: `MissionSynthesis` class — init / add_event / combine

**Files:**
- Modify: `FatigueDS/mission_synthesis.py`
- Test: `tests/test_mission_synthesis.py`

**Interfaces:**
- Consumes: `combine_fds`, `envelope_ers` (Task 1); `Spectrum` objects carrying `.f0_range`, `.fds`, `.ers`.
- Produces: `MissionSynthesis` with `__init__(events=None)`, `add_event(spectrum, repeats=1)`, `combine()` (sets `self.f0_range`, `self.fds_ref`, `self.ers_ref`); `self.events` is a list of `(spectrum, repeats)` tuples.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_synthesis.py`:

```python
from FatigueDS.mission_synthesis import MissionSynthesis


def _make_event(freq_range=(20, 200, 5), amp=10.0):
    s = FatigueDS.Spectrum(freq_data=freq_range)
    s.set_sine_load(sine_freq=100, amp=amp, t_total=3600)
    s.get_ers()
    s.get_fds(k=5, C=1, p=1)
    return s


def test_combine_two_events_sum_and_envelope():
    e1 = _make_event(amp=10.0)
    e2 = _make_event(amp=10.0)
    ms = MissionSynthesis([e1, e2])
    ms.combine()
    assert np.allclose(ms.fds_ref, 2 * e1.fds)
    assert np.allclose(ms.ers_ref, np.maximum(e1.ers, e2.ers))
    assert np.allclose(ms.f0_range, e1.f0_range)


def test_combine_repeats():
    e1 = _make_event(amp=10.0)
    ms = MissionSynthesis()
    ms.add_event(e1, repeats=3)
    ms.combine()
    assert np.allclose(ms.fds_ref, 3 * e1.fds)


def test_combine_f0_mismatch_raises():
    e1 = _make_event(freq_range=(20, 200, 5))
    e2 = _make_event(freq_range=(20, 300, 5))
    ms = MissionSynthesis([e1, e2])
    with pytest.raises(ValueError):
        ms.combine()


def test_add_event_bad_repeats():
    ms = MissionSynthesis()
    with pytest.raises(ValueError):
        ms.add_event(_make_event(), repeats=0)


def test_combine_missing_fds_raises():
    s = FatigueDS.Spectrum(freq_data=(20, 200, 5))
    s.set_sine_load(sine_freq=100, amp=10.0, t_total=3600)
    s.get_ers()  # no get_fds -> no .fds
    ms = MissionSynthesis([s])
    with pytest.raises(ValueError):
        ms.combine()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission_synthesis.py -k "combine_two or repeats or mismatch or missing_fds or bad_repeats" -v`
Expected: FAIL — `ImportError: cannot import name 'MissionSynthesis'`

- [ ] **Step 3: Write minimal implementation**

Append to `FatigueDS/mission_synthesis.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mission_synthesis.py -k "combine_two or repeats or mismatch or missing_fds or bad_repeats" -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add FatigueDS/mission_synthesis.py tests/test_mission_synthesis.py
git commit -m "feat: add MissionSynthesis combine (FDS summation, ERS envelope)"
```

---

### Task 4: `MissionSynthesis.invert`

**Files:**
- Modify: `FatigueDS/mission_synthesis.py`
- Test: `tests/test_mission_synthesis.py`

**Interfaces:**
- Consumes: `invert_fds_to_psd` (Task 2); event attributes `.k`, `.C`, `.p` (set by `get_fds`), `.Q` (set by `Spectrum.__init__`); `self.fds_ref`, `self.f0_range` (Task 3).
- Produces: `invert(T_test, k=None, C=None, p=None, Q=None)` setting `self.test_psd`, `self.test_psd_freq` (= `self.f0_range`), and storing `self.k, self.C, self.p, self.Q, self.T_test`; private `_resolve_param(name, override)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_synthesis.py`:

```python
def test_invert_uses_event_params_and_matches_helper():
    e1 = _make_event(amp=10.0)  # built with k=5, C=1, p=1, Q=10 (default)
    ms = MissionSynthesis([e1])
    ms.combine()
    ms.invert(T_test=1800.0)
    expected = invert_fds_to_psd(ms.fds_ref, ms.f0_range, k=5, C=1, p=1, Q=10, T_test=1800.0)
    assert np.allclose(ms.test_psd, expected)
    assert np.allclose(ms.test_psd_freq, ms.f0_range)


def test_invert_before_combine_raises():
    ms = MissionSynthesis([_make_event()])
    with pytest.raises(ValueError):
        ms.invert(T_test=1800.0)


def test_invert_bad_T_test_raises():
    ms = MissionSynthesis([_make_event()])
    ms.combine()
    with pytest.raises(ValueError):
        ms.invert(T_test=0)


def test_invert_inconsistent_params_raises():
    e1 = _make_event(amp=10.0)
    e2 = _make_event(amp=10.0)
    e2.k = 7  # force inconsistent S-N slope across events
    ms = MissionSynthesis([e1, e2])
    ms.combine()
    with pytest.raises(ValueError):
        ms.invert(T_test=1800.0)


def test_invert_override_param():
    e1 = _make_event(amp=10.0)
    e2 = _make_event(amp=10.0)
    e2.k = 7
    ms = MissionSynthesis([e1, e2])
    ms.combine()
    ms.invert(T_test=1800.0, k=5)  # override resolves the inconsistency
    assert ms.k == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission_synthesis.py -k "invert_uses or invert_before or bad_T_test or inconsistent or override" -v`
Expected: FAIL — `AttributeError: 'MissionSynthesis' object has no attribute 'invert'`

- [ ] **Step 3: Write minimal implementation**

Append the two methods to the `MissionSynthesis` class in `FatigueDS/mission_synthesis.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mission_synthesis.py -k "invert_uses or invert_before or bad_T_test or inconsistent or override" -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add FatigueDS/mission_synthesis.py tests/test_mission_synthesis.py
git commit -m "feat: add MissionSynthesis.invert (reference FDS -> test PSD)"
```

---

### Task 5: `MissionSynthesis.check_ers`

**Files:**
- Modify: `FatigueDS/mission_synthesis.py`
- Test: `tests/test_mission_synthesis.py`

**Interfaces:**
- Consumes: `Spectrum` (for ERS of the derived PSD); `self.test_psd`, `self.test_psd_freq`, `self.ers_ref`, `self.Q`, `self.T_test` (Tasks 3–4).
- Produces: `check_ers(tol=0.05)` setting `self.ers_test` and `self.ers_ratio` (= `ers_test/ers_ref`, 0 where `ers_ref==0`); emits `warnings.warn` when `max(ers_ratio) > 1 + tol`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_synthesis.py`:

```python
def _random_event(T):
    rng = np.random.default_rng(0)
    fs = 2048.0
    acc = rng.standard_normal(int(T * fs))
    s = FatigueDS.Spectrum(freq_data=(50, 400, 10), Q=10)
    s.set_random_load((acc, 1 / fs), unit='ms2')
    s.get_ers()
    s.get_fds(k=8, C=1, p=1)
    return s


def test_check_ers_sets_ratio():
    ms = MissionSynthesis([_random_event(T=20.0)])
    ms.combine()
    ms.invert(T_test=20.0)  # same duration -> ERS ~ reference
    ms.check_ers()
    assert ms.ers_ratio.shape == ms.f0_range.shape
    assert np.all(np.isfinite(ms.ers_ratio))


def test_check_ers_warns_on_overtest():
    ms = MissionSynthesis([_random_event(T=20.0)])
    ms.combine()
    ms.invert(T_test=2.0)  # 10x acceleration -> higher PSD -> higher test ERS
    with pytest.warns(UserWarning):
        ms.check_ers()


def test_check_ers_before_invert_raises():
    ms = MissionSynthesis([_random_event(T=20.0)])
    ms.combine()
    with pytest.raises(ValueError):
        ms.check_ers()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission_synthesis.py -k "check_ers" -v`
Expected: FAIL — `AttributeError: 'MissionSynthesis' object has no attribute 'check_ers'`

- [ ] **Step 3: Write minimal implementation**

Add the import near the top of `FatigueDS/mission_synthesis.py` (below the existing imports):

```python
import warnings
from .spectrum import Spectrum
```

Append the method to the `MissionSynthesis` class:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mission_synthesis.py -k "check_ers" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add FatigueDS/mission_synthesis.py tests/test_mission_synthesis.py
git commit -m "feat: add MissionSynthesis.check_ers over-test guard"
```

---

### Task 6: Plotting methods + package export

**Files:**
- Modify: `FatigueDS/mission_synthesis.py`
- Modify: `FatigueDS/__init__.py`
- Test: `tests/test_mission_synthesis.py`

**Interfaces:**
- Consumes: `self.f0_range`, `self.fds_ref`, `self.ers_ref`, `self.test_psd` (Tasks 3–4).
- Produces: `plot_fds(new_figure=True, **kwargs)`, `plot_ers(new_figure=True, **kwargs)`, `plot_test_psd(new_figure=True, **kwargs)`; `MissionSynthesis`, `combine_fds`, `envelope_ers`, `invert_fds_to_psd` importable from the top-level `FatigueDS` package.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mission_synthesis.py`:

```python
import matplotlib
matplotlib.use('Agg')


def test_exports_from_package():
    assert hasattr(FatigueDS, 'MissionSynthesis')
    assert hasattr(FatigueDS, 'invert_fds_to_psd')


def test_plots_run():
    ms = MissionSynthesis([_make_event()])
    ms.combine()
    ms.invert(T_test=1800.0)
    ms.plot_fds()
    ms.plot_ers()
    ms.plot_test_psd()


def test_plot_before_combine_raises():
    ms = MissionSynthesis([_make_event()])
    with pytest.raises(ValueError):
        ms.plot_fds()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mission_synthesis.py -k "exports or plots_run or plot_before" -v`
Expected: FAIL — `AttributeError: module 'FatigueDS' has no attribute 'MissionSynthesis'`

- [ ] **Step 3: Write minimal implementation**

Add the matplotlib import near the top of `FatigueDS/mission_synthesis.py`:

```python
import matplotlib.pyplot as plt
```

Append the three methods to the `MissionSynthesis` class:

```python
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
```

Modify `FatigueDS/__init__.py` to add (after the existing `from .spectrum import Spectrum` line):

```python
from .mission_synthesis import (
    MissionSynthesis,
    combine_fds,
    envelope_ers,
    invert_fds_to_psd,
)
```

- [ ] **Step 4: Run the full test suite to verify everything passes**

Run: `python -m pytest tests/test_mission_synthesis.py -v`
Expected: PASS (all tests in the file pass)

- [ ] **Step 5: Commit**

```bash
git add FatigueDS/mission_synthesis.py FatigueDS/__init__.py tests/test_mission_synthesis.py
git commit -m "feat: add MissionSynthesis plotting and package exports"
```

---

## Self-Review

**Spec coverage:**
- Combine FDS (summation) → Task 1 (`combine_fds`) + Task 3 (`combine`). ✓
- Combine ERS (envelope) → Task 1 (`envelope_ers`) + Task 3. ✓
- Invert FDS → test PSD (eq [4.9] inverse) → Task 2 (`invert_fds_to_psd`) + Task 4 (`invert`). ✓
- Over-test ERS guard (warning) → Task 5 (`check_ers`). ✓
- `MissionSynthesis` class + pure helpers → Tasks 1–6. ✓
- Shared `f0_range` validation, missing `.fds`/`.ers`, inconsistent params, `repeats<=0`, `T_test<=0`, ordering guards → Tasks 3–6. ✓
- Plotting + exports → Task 6. ✓
- Round-trip closure test → Task 2. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; every test step has real assertions. ✓

**Type consistency:** `invert_fds_to_psd(fds_ref, f0_range, k, C, p, Q, T_test)` signature is identical in Tasks 2 and 4. `combine_fds(fds_list, repeats)` / `envelope_ers(ers_list)` consistent between Tasks 1 and 3. Class attributes (`fds_ref`, `ers_ref`, `f0_range`, `test_psd`, `test_psd_freq`, `ers_ratio`, `k/C/p/Q/T_test`) named consistently across Tasks 3–6. ✓
