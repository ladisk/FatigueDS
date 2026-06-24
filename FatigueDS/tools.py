"""Helper functions for the FatigueDS :class:`~FatigueDS.spectrum.Spectrum` class:
frequency-range handling, the SDOF response RMS over a segmented PSD, the closed-form
spectral integrals, the time-domain SDOF response, and material-parameter conversion.

References
----------
.. [Lalanne3] C. Lalanne, *Mechanical Vibration and Shock Analysis, Vol. 3:
   Random Vibration*, 2nd ed., ISTE/Wiley, 2009.
.. [Lalanne4] C. Lalanne, *Mechanical Vibration and Shock Analysis, Vol. 4:
   Fatigue Damage*, 2nd ed., ISTE/Wiley, 2009.
.. [Lalanne5] C. Lalanne, *Mechanical Vibration and Shock Analysis, Vol. 5:
   Specification Development*, 2nd ed., ISTE/Wiley, 2009.
.. [Thomson] W. T. Thomson, *Theory of Vibration with Applications*, 2nd ed.,
   Prentice-Hall, 1981.
"""

import numpy as np
from scipy import signal
from FLife.tools import basquin_to_sn

def convert_Q_damp(self, Q=None, damp=None): 
    """
    Function for converting damping ratio to Q-factor and vice versa.

    :param Q: damping Q-factor [/]
    :param damp: damping ratio [/]
    """

    if damp is not None:
        self.damp = damp
        self.Q = 1 / (2 * self.damp)

    elif Q is not None:
        self.Q = Q
        self.damp = 1 / (2 * self.Q)

def get_freq_range(self, freq_data):
    """
    Function for generating frequency ranges-> X-axis of MRS/FDS plot from freq_data tuple.

    :param freq_data: frequency data (tuple)

    :return: frequency range
    """
    if isinstance(freq_data, tuple) and len(freq_data) == 3:
        f0_start, f0_stop, f0_step = freq_data
        f0_range = np.arange(f0_start, f0_stop + f0_step, f0_step, dtype=float)
    else:
        f0_range = freq_data       
    
    if f0_range[0] == 0:
        f0_range[0] = 1e-3    # sets frequency to a small number to avoid dividing by 0
    
    return f0_range


def rms_sum(f_0, psd_freq, psd_data, damp, motion='rel_disp'):
    """
    This function calculates the response RMS (either relative displacement, velocity or acceleration) for a given
    natural frequency and damping ratio.

    The excitation PSD is treated as ``n`` horizontal straight-line segments and the
    response mean square is accumulated segment by segment using the closed-form
    integrals I0/I2/I4 (:func:`integrals_b`). This is the summation kernel of
    [Lalanne3]_ p.395, eq. [8.86]; the ``pi/(4*xi)`` and ``f0``-power prefactors are
    applied by the caller in ``spectrum.py``.

    :param f_0: system natural frequency [Hz]
    :param psd_freq: PSD frequency range [Hz]
    :param psd_data: PSD data [(m/s^2)^2/Hz] or [g^2/Hz]
    :param damp: damping ratio [/]
    :param motion: which rms sum to perform (supported: rel_disp, rel_vel and rel_acc)

    :return: RMS sum value
    """
    
    df = np.diff(psd_freq)[0]
    rms_sum = 0

    f1 = psd_freq - df / 2
    f2 = psd_freq + df / 2

    # Adjust first and last elements for f1 and f2 respectively
    f1[0] = psd_freq[0]
    f2[-1] = psd_freq[-1]

    for j in range(len(psd_data)):

        h1 = f1[j] / f_0
        h2 = f2[j] / f_0

        # Per-segment contribution G_j * [I_b(h2) - I_b(h1)], the summation kernel of
        # [Lalanne3] p.395 eq [8.86] (rel_disp uses I0, rel_vel uses I2, rel_acc uses I4).
        if motion == 'rel_disp':
            z_rms = psd_data[j] * (integrals_b(h=h2, b=0, damp=damp) - integrals_b(h=h1, b=0, damp=damp))
            rms_sum += z_rms

        elif motion == 'rel_vel':
            dz_rms = psd_data[j] * (integrals_b(h=h2, b=2, damp=damp) - integrals_b(h=h1, b=2, damp=damp))
            rms_sum += dz_rms
            
        elif motion == 'rel_acc':
            ddz_rms = psd_data[j] * (integrals_b(h=h2, b=4, damp=damp) - integrals_b(h=h1, b=4, damp=damp))
            rms_sum += ddz_rms
       
    return rms_sum



def integrals_b(h, b, damp):
    """
    Closed-form evaluation of the dimensionless spectral integrals I0, I2 and I4 used in
    the segmented-PSD response RMS (:func:`rms_sum`, [Lalanne3]_ eq. [8.86]).

    The closed forms are [Lalanne3]_ Appendix A6, p.544: I0 = eq. [A6.20], I2 = eq.
    [A6.22], I4 = eq. [A6.24]. They are identical to eqs. (A1-74), (A1-75), (A1-76) in
    [Lalanne5]_. ``alpha = 2*sqrt(1-xi**2)`` and ``beta = 2*(1-2*xi**2)`` ([Lalanne3]_
    eq. [8.37]).

    :param h: frequency ratio (frequency vs natural frequency) [/]
    :param b: exponent b [/] (0, 2 or 4)
    :param damp: damping ratio [/]

    :return: I_b integral value
    """

    # constants ([Lalanne3] eq [8.37])
    alpha = 2 * np.sqrt(1 - damp**2)
    beta = 2 * (1 - 2 * damp**2)

    C0 = damp / (np.pi * alpha)
    C1 = (h**2 + alpha * h + 1)/(h**2 - alpha * h + 1)
    C2 = (2 * h + alpha) / (2 * damp)
    C3 = (2 * h - alpha) / (2 * damp)
    C4 = 4 * damp / np.pi
    C5 = np.arctan(C2) + np.arctan(C3)

    # integrals
    if b == 0:
        Ib = C0 * np.log(C1) + 1 / np.pi * C5  # I0: [Lalanne3] p.544 eq [A6.20] (= [Lalanne5] eq (A1-74))

    elif b == 2:
        Ib = C0*np.log(1 / C1) + 1 / np.pi * C5  # I2: [Lalanne3] p.544 eq [A6.22] (= [Lalanne5] eq (A1-75)); log argument reciprocated vs I0

    elif b == 4:
        I0 = C0 * np.log(C1) + 1 / np.pi * C5
        I2 = -C0 * np.log(C1) + 1 / np.pi * C5
        Ib = C4 * h + beta * I2 - I0  # I4: [Lalanne3] p.544 eq [A6.24] (= [Lalanne5] eq (A1-76))

    else:
        raise ValueError(f"Invalid exponent ``b``='{b}'. Supported exponents: 0, 2 and 4.")
    
    return Ib


def response_relative_displacement(time_data, dt, f_0, damp):
    """
    Returns the relative response displacement of a linear SDOF system by convolving the
    base-excitation signal with the system's unit-impulse response (Duhamel's integral).
    Used to obtain the SDOF response for the time-domain ERS/FDS of a random signal.

    The impulse response used here is that of the *damped* SDOF system,
    ``h(t) = -1/omega_d * exp(-xi*omega_0*t) * sin(omega_d*t)`` with
    ``omega_d = omega_0*sqrt(1-xi**2)`` -- the damped generalisation of the impulse
    response / convolution (Duhamel) integral in [Thomson]_ ch. 4 (impulse response and
    convolution, eq. (4.2-5) gives the undamped case). The leading sign and the
    ``1/omega_d`` factor give the relative displacement of the mass w.r.t. its base for a
    unit base acceleration.

    :param time_data: signal time data [m/s^2]
    :param dt: time step [s]
    :param f_0: system natural frequency [Hz]
    :param damp: damping ratio [/]

    :return: relative response displacement [m]
    """
    n = len(time_data)
    time = np.arange(n) * dt
    
    omega_0 = 2 * np.pi * f_0
    omega_0d = omega_0 * np.sqrt(1 - damp**2)
    
    impulse_resp_func = -1 / omega_0d * np.exp(-damp * omega_0 * time) * np.sin(omega_0d * time)

    z = signal.convolve(time_data, impulse_resp_func)[:len(time)] * dt
    
    return z


def psd_averaging(self):
    """
    PSD averaging method: Welch's method for calculating PSD of a random signal frm time data.
    """

    if not hasattr(self, 'bins'):
        raise ValueError('Number of bins ``bins`` must be provided for PSD averaging method.')
    
    freq_avg, psd_avg = signal.welch(
        self.time_data, 
        fs=1 / self.dt, 
        nperseg=len(self.time_data) // self.bins, 
        window='boxcar', 
        scaling='density',
        )
    
    self.psd_data = psd_avg
    self.psd_freq = freq_avg

def material_parameters_convert(sigma_f, b, range = False):
    """
    Converts Basquin equation parameters ``sigma_f`` and ``b`` to fatigue life parameters ``C`` and ``k``,
    using a function from FLife package. Basic form of Basquin equation is used here: ``sigma_a = sigma_f* (2*N)**b``. The function converts to parameters from equation ``N * s**k = C``
    (the S-N / Basquin law, [Lalanne4]_ p.22, eqs. [1.13]-[1.15]).

    :param sigma_f:
        Fatigue strength coefficient [MPa**k].
    :param b:
        Fatigue strength exponent [/]. Represents S-N curve slope.
    :param range:
        False/True sets returned value C with regards to amplitude / range count, respectively.
    
    :return C,k:
        C - S-N curve intercept [MPa**k], k - S-N curve inverse slope [/].

    """

    C,k = basquin_to_sn(sigma_f, b, range=range)
    
    return C, k 
