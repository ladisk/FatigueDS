# Mission Synthesis — design spec

Status: approved for planning · Date: 2026-06-22 · Branch: `feature/mission-synthesis`

## 1. Purpose

Add the Lalanne "specification development" workflow to FatigueDS: take the fatigue
damage of several real-life vibration events, combine them into a reference FDS (and
ERS), and **invert** that reference into an equivalent accelerated **test PSD** for a
chosen test duration — with an over-test guard against the reference ERS.

All methods follow Christian Lalanne, *Mechanical Vibration and Shock Analysis*, 2nd
ed., Wiley-ISTE (2009), Vol. 4 (Fatigue Damage) and Vol. 5 (Specification
Development). Notation maps Lalanne `b, K` to the package's `k, p` (with `C`
unchanged), consistent with `Spectrum.get_fds`.

## 2. Goals / non-goals

**Goals**
- Combine the FDS/ERS of multiple `Spectrum` events into reference curves.
- Invert the reference FDS to a test PSD for a user-chosen test duration `T_test`
  (accelerated testing: shorter duration → higher level).
- Compare the resulting test ERS to the reference ERS and warn on over-test.
- A `MissionSynthesis` class plus pure, separately-testable helper functions.

**Non-goals (explicitly out; candidates for later steps)**
- Test / uncertainty / aging factors and statistical guard coefficients.
- Matrix-inversion method (Vol. 5 §11.4.9.1, eq [11.2]) — Lalanne himself recommends
  the iteration method (§11.4.9.2, eq [11.11]) over it (instability); the iteration
  method IS implemented (see §3.3).
- Duration-vs-level optimization / exaggeration-factor search.
- New event input formats — events are `Spectrum` objects (reuse the existing class).

## 3. Theory and formulas

Symbols: `f0` natural frequency, `ω0 = 2πf0`, `Q` quality factor, `k` S-N slope
(Basquin `N·s^k = C`), `p` stress↔relative-displacement constant, `C` S-N constant,
`T` duration, `Γ` gamma function. Reference FDS denoted `D_ref(f0)`.

**3.1 Combine FDS — damage summation** (fatigue accumulates across events):

    fds_ref(f0) = Σ_i  repeats_i · fds_i(f0)

**3.2 Combine ERS — max-envelope** (extreme stress is the worst single event):

    ers_ref(f0) = max_i  ers_i(f0)      (elementwise)

**3.3 Invert FDS → test PSD.** Two methods.

*(a) Iteration method (default — Lalanne Vol. 5 §11.4.9.2, eq [11.11]).* Start from an
initial PSD and rescale each level by the target/achieved FDS ratio:

    G_{n+1}(f0) = G_n(f0) · ( D_ref(f0) / FDS(G_n)(f0) )^(2/k)

where `FDS(G_n)` is the *full forward* FDS of the candidate PSD (computed with
`Spectrum`, so it includes each oscillator's off-resonance response). Iterate, keep
the best-fitting PSD, and stop when it stops improving (patience). This reproduces the
reference FDS to a few % (median ~0) — far better than (b), which ignores off-resonance
coupling and can be >100% off for a stepped PSD. A summed/enveloped reference FDS is
generally not the exact FDS of any single PSD (FDS is non-linear in the PSD for k≠2),
so a small irreducible residual at sharp features is expected.

*(b) Closed-form (option `method='closed_form'`)* — fast diagonal/narrow-band inverse
of Lalanne's white-noise FDS (Vol. 5 eq [4.9]; the forward is validated in
`validation/validation.md` to <0.05% at Q=200). Used as the iteration's warm start:

forward:  `D = (p^k/C) · f0 · T · (Q·G/(2·ω0³))^(k/2) · Γ(1+k/2)`

inverse:

    G(f0) = (2·ω0³ / Q) · [ C · D_ref(f0) / (p^k · f0 · T_test · Γ(1+k/2)) ]^(2/k)

`G(f0)` is the acceleration PSD level on the `f0_range` grid (units (m/s²)²/Hz).
Where `D_ref(f0) = 0` or `f0 ≈ 0`, set `G(f0) = 0` (avoid division blow-up). Matrix
inversion (eq [11.2]) is intentionally not implemented — Lalanne prefers the iteration.

**3.4 Over-test guard:** construct a `Spectrum` from `(test_psd, f0_range)` with
`T = T_test`, `Q`, compute its ERS, and report

    ers_ratio(f0) = ers_test(f0) / ers_ref(f0)

Warn (do not raise) where `ers_ratio > 1 + tol` (default `tol = 0.05`): the
accelerated test exceeds field extreme response and may excite field-absent failure
modes.

Note: `check_ers` feeds `test_psd` back through `Spectrum`'s random-PSD route, which
treats the PSD as piecewise-constant bins of width `df = diff(f0_range)[0]`; it
therefore assumes a **uniformly spaced** `f0_range` (the common case). A non-uniform
grid is out of scope here.

## 4. API

New module `FatigueDS/mission_synthesis.py`; `MissionSynthesis` exported from
`FatigueDS/__init__.py`. Pure helpers also exported for direct use.

```python
class MissionSynthesis:
    def __init__(self, events=None):
        """events: optional list of Spectrum (each with .fds/.ers on a shared f0_range)."""

    def add_event(self, spectrum, repeats=1):
        """Append a Spectrum event with an occurrence multiplier (repeats > 0)."""

    def combine(self):
        """Set self.fds_ref (Σ repeats·fds), self.ers_ref (elementwise max), self.f0_range."""

    def invert(self, T_test, k=None, C=None, p=None, Q=None,
               method='iteration', max_iter=200, warn_tol=0.25):
        """Set self.test_psd (G on grid) and self.test_psd_freq = self.f0_range.
        method='iteration' (default, eq [11.11]) or 'closed_form' (diagonal, eq [4.9]).
        For 'iteration' also sets self.invert_n_iter and self.invert_error; warns if the
        achieved FDS error exceeds warn_tol. Material params default to the
        (validated-consistent) event values; args override."""

    def check_ers(self, tol=0.05):
        """Set self.ers_ratio; warn where test ERS exceeds reference ERS by > tol."""

    def plot_fds(self, **kw); def plot_ers(self, **kw); def plot_test_psd(self, **kw)
```

Pure helper functions (module level):

```python
def combine_fds(fds_list, repeats=None) -> np.ndarray      # damage summation
def envelope_ers(ers_list) -> np.ndarray                   # elementwise max
def invert_fds_to_psd(fds_ref, f0_range, k, C, p, Q, T_test) -> np.ndarray  # eq [4.9] inverse (diagonal)
def invert_fds_to_psd_iterative(fds_ref, f0_range, k, C, p, Q, T_test,
                                max_iter=200, patience=15) -> (np.ndarray, int, float)  # eq [11.11]
```

State set by the workflow: `f0_range`, `fds_ref`, `ers_ref`, `test_psd`,
`test_psd_freq`, `ers_ratio`, plus stored `k, C, p, Q, T_test` (and, for the iteration
method, `invert_n_iter`, `invert_error`).

## 5. Data flow

    events (Spectrum, shared f0_range, each with .fds/.ers)
       → combine()        → fds_ref, ers_ref
       → invert(T_test)   → test_psd (G on f0_range)
       → check_ers()      → ers_ratio (+ over-test warning)

## 6. Error handling (raise ValueError unless noted)

- Events do not share an identical `f0_range` (compared with `np.allclose`).
- An event is missing `.fds` (for combine/invert) or `.ers` (for combine ERS).
- Material params `k/C/p/Q` differ across events and no override is given to `invert`.
- `repeats <= 0` or `T_test <= 0`.
- `combine()` not called before `invert()`, or `invert()` not called before `check_ers()`.
- Over-test is a **warning** (`warnings.warn`), not an error.

## 7. Testing (literature-anchored)

- **Round-trip closure (primary):** a flat PSD over a **wide** band → `Spectrum.get_fds`
  → `invert_fds_to_psd` at the same `T_test` → recovered PSD matches the original at
  **mid-band** f0 within narrow-band tolerance (the forward path uses the full spectral
  integral; the inverse uses the narrow-band [4.9], so closure is exact only for
  high-Q, mid-band — use e.g. Q≥25 and a ~5–10% tolerance). Validates the inverse
  against the package's own forward computation.
- **Combine:** two identical events → `fds_ref = 2×` single; `repeats=2` on one event
  gives the same; `ers_ref` = elementwise max of inputs.
- **Acceleration:** inverting the same reference at `T_test = T/10` gives a PSD a factor
  `10^(2/k)` higher, and `check_ers` flags the elevated ERS.
- **Errors:** each guard in §6.

## 8. Files

- `FatigueDS/mission_synthesis.py` — new.
- `FatigueDS/__init__.py` — export `MissionSynthesis` (and helpers).
- `tests/test_mission_synthesis.py` — new test module.
- `docs/` — a short usage page and an example (sequenced with the package's existing
  Sphinx docs) — optional, can follow implementation.
