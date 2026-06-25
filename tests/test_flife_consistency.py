"""Cross-checks that tie FatigueDS to the FLife package.

The random-PSD FDS of FatigueDS and the narrow-band damage of FLife are the *same* closed
form (Lalanne narrow-band Rayleigh = Miles / Bendat-Piersol):

    FatigueDS  D(f0) = (p**k / C) * n0 * T * (sqrt(2) * z_rms)**k * Gamma(1 + k/2)
    FLife      D     =              nu * T * (sqrt(2 * m0))**k     * Gamma(1 + k/2) / C

so with the per-oscillator stress sigma = p * z (=> m0 = (p * z_rms)**2) and nu = n0 they are
identical. These tests verify that identity numerically and exercise the FLife interop
helpers. FLife (and its optional GUI dependencies) is required; the module is skipped if it
cannot be imported.
"""
import numpy as np
import pytest

pytest.importorskip("FLife")
import FLife  # noqa: E402
import FatigueDS  # noqa: E402


def _sdof_stress_response_psd(f0, psd, freq, Q, p):
    """Stress PSD of the SDOF relative-displacement response to a base-acceleration PSD.

    |H_z(f)|**2 = 1 / [omega0**4 * ((1 - r**2)**2 + (r / Q)**2)],  r = f / f0,
    and stress = p * z, so the stress PSD is p**2 * |H_z|**2 * psd.
    """
    w0 = 2 * np.pi * f0
    r = freq / f0
    Hz2 = 1.0 / (w0**4 * ((1 - r**2)**2 + (r / Q)**2))
    return (p**2) * Hz2 * psd


def test_fds_matches_flife_narrowband():
    """FatigueDS FDS(f0) equals the FLife Narrowband damage of the oscillator stress response."""
    T, C, p = 3600.0, 1.0, 1.0
    Q, k = 20, 6
    freq = np.arange(0, 2000, 0.25)
    psd = np.where((freq >= 20) & (freq <= 1000), 1.0, 0.0)

    for f0 in (150.0, 300.0, 500.0):
        s = FatigueDS.Spectrum(freq_data=np.array([f0]), Q=Q)
        s.set_random_load((psd, freq), unit='ms2', T=T)
        s.get_fds(k=k, C=C, p=p)
        fds = s.fds[0]

        g_stress = _sdof_stress_response_psd(f0, psd, freq, Q, p)
        sd = FLife.SpectralData(input={'PSD': g_stress, 'f': freq})
        damage = T / FLife.Narrowband(sd).get_life(C, k)

        assert np.isclose(fds, damage, rtol=2e-3), (f0, fds, damage)


def test_set_random_load_accepts_spectraldata():
    """A FLife.SpectralData can be passed straight into set_random_load (PSD interop)."""
    freq = np.arange(0, 2000, 1.0)
    psd = np.where((freq >= 50) & (freq <= 800), 2.0, 0.0)
    fr = (20, 1000, 20)
    k, C, p = 6, 1.0, 1.0

    ref = FatigueDS.Spectrum(freq_data=fr, Q=10)
    ref.set_random_load((psd, freq), unit='ms2', T=100.0)
    ref.get_ers()
    ref.get_fds(k=k, C=C, p=p)

    sd = FLife.SpectralData(input={'PSD': psd, 'f': freq})
    via = FatigueDS.Spectrum(freq_data=fr, Q=10)
    via.set_random_load(sd, unit='ms2', T=100.0)
    via.get_ers()
    via.get_fds(k=k, C=C, p=p)

    assert np.allclose(ref.ers, via.ers)
    assert np.allclose(ref.fds, via.fds)


def test_mission_synthesis_to_flife_input():
    """The mission-synthesis test PSD round-trips into FLife for life estimation."""
    fr = (20, 600, 10)
    freq = np.arange(20, 600, 1.0)
    k, C, p, Q = 7, 1.0, 1.0, 10
    psd = 1.0 * np.exp(-0.5 * ((freq - 200) / 60)**2)

    ev = FatigueDS.Spectrum(freq_data=fr, Q=Q)
    ev.set_random_load((psd, freq), unit='ms2', T=3600.0)
    ev.get_ers()
    ev.get_fds(k=k, C=C, p=p)

    ms = FatigueDS.MissionSynthesis()
    ms.add_event(ev, repeats=2)
    ms.combine()
    ms.invert(T_test=600.0)

    flinput = ms.to_flife_input()
    assert set(flinput) == {'PSD', 'f'}

    sd = FLife.SpectralData(input=flinput)
    life = FLife.Narrowband(sd).get_life(C, k)
    assert np.isfinite(life) and life > 0


@pytest.mark.parametrize("range_flag", [False, True])
def test_material_parameter_conversion_matches_flife_and_roundtrips(range_flag):
    """material_parameters_convert matches FLife's basquin_to_sn, and the new inverse round-trips."""
    from FLife.tools import basquin_to_sn

    sigma_f, b = 800.0, -0.1
    C, k = FatigueDS.tools.material_parameters_convert(sigma_f, b, range=range_flag)
    C_ref, k_ref = basquin_to_sn(sigma_f, b, range=range_flag)
    assert np.isclose(C, C_ref) and np.isclose(k, k_ref)

    sigma_f_back, b_back = FatigueDS.tools.material_parameters_convert_to_basquin(
        C, k, range=range_flag)
    assert np.isclose(sigma_f_back, sigma_f) and np.isclose(b_back, b)
