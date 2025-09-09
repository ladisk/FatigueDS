import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.special import gamma
import rainflow
from tqdm import tqdm

from . import tools

class Spectrum:

    def __init__(self, freq_data=(10, 2000, 5), damp=None, Q=10):
        """
        Initialize the Spectrum class. Frequency range and damping ratio/Q-factor must be provided.
        Only one of the damping ratio or Q-factor must be provided. If both are provided, damping ratio will be used. If None, Q=10 will be used.

        :param freq_data: tuple containing (f0_start, f0_stop, f0_step) [Hz] or a frequency vector, defining the range where the ERS and FDS will be calculated
        :param damp: damping ratio [/]
        :param Q: damping Q-factor [/] (default: Q=10)
        """

        # check freq_data input
        if (isinstance(freq_data, tuple) and len(freq_data) == 3) or (
            isinstance(freq_data, np.ndarray) and freq_data.ndim == 1
        ):
           self.f0_range = tools.get_freq_range(self, freq_data)
        else:
            raise ValueError('``f0`` should be a tuple containing (f0_start, f0_stop, f0_step) [Hz] or a frequency vector')
        
        # check damping input (Q or damp)
        if isinstance(damp, (int, float)) or isinstance(Q, (int, float)):
            tools.convert_Q_damp(self, Q=Q, damp=damp)


    def set_sine_load(self, sine_freq=None, amp=None, t_total=None, exc_type='acc', unit='ms2'):
        """
        Set sine signal load parameters

        :param sine_freq: sine frequency [Hz]
        :param amp: signal amplitude [m/s^2, m/s, m]
        :param t_total: total time duration of the signal [s] (only needed for fds calculation)
        :param exc_type: excitation type (supported: 'acc [m/s^2]', 'vel[m/s]' and 'disp[m]')
        :param unit: unit of the signal (supported: 'g' and 'ms2') Parameter only needed for fds calculation
        """

        self.signal_type = 'sine'

        if all([sine_freq, amp, exc_type]):
            self.sine_freq = sine_freq
            self.amp = amp
            self.exc_type = exc_type
        else:    
            raise ValueError('Missing parameter(s). ``sine_freq`` and ``amp`` must be provided')
        
        if isinstance(t_total, (int, float)):
            self.t_total = t_total

            
        if self.exc_type in ['acc', 'vel', 'disp']:            
            if self.exc_type == 'acc':
                self.a = 0
            elif self.exc_type == 'vel':
                self.a = 1
            elif self.exc_type == 'disp':
                self.a = 2
        else:
            raise ValueError(f"Invalid excitation type. Supported types: ``acc``, ``vel`` and ``disp``.")
        
        if unit == 'g':
            self.unit_scale = 9.81
        elif unit == 'ms2':
            self.unit_scale = 1
        else:
            raise ValueError("Invalid unit selected. Supported units: 'g' and 'ms2'.")



    def set_sine_sweep_load(self, const_amp=None, const_f_range=None, exc_type='acc', dt=1, sweep_type=None, sweep_rate=None, unit='ms2'):
        """
        Set sine sweep signal load parameters
        
        :param const_amp: constant amplitude ranges  [m/s^2, m/s, m]
        :param const_f_range: constant frequency ranges [Hz]
        :param exc_type: excitation type (supported: 'acc [m/s^2]', 'vel[m/s]' and 'disp[m]')
        :param dt: time step [s] (default 1 second)
        :param sweep_type: sine sweep type (['linear','lin'] or ['logarithmic','log']) 
        :param sweep_rate: sinusoidal sweep rate [Hz/min] for 'linear' and [oct./min] for 'logarithmic' sweep type
        :param unit: unit of the signal (supported: 'g' and 'ms2') Parameter only needed for fds calculation
        """
        
        self.signal_type = 'sine_sweep'
        if None not in [const_amp, const_f_range, exc_type, dt, sweep_type, sweep_rate]:
            # necessary parameters
            self.const_amp = const_amp
            self.const_f_range = const_f_range
            self.sweep_type = sweep_type
            self.sweep_rate = sweep_rate
            # optional parameters
            self.exc_type = exc_type
            self.dt = dt
        else:
            raise ValueError('Missing parameter(s). ``const_amp``, ``const_f_range``, ``sweep_type`` and ``sweep_rate`` must be provided')

        if self.exc_type in ['acc','vel','disp']:   
            if self.exc_type == 'acc':
                self.a = 0
            elif self.exc_type == 'vel':
                self.a = 1
            elif self.exc_type == 'disp':
                self.a = 2
        else:
            raise ValueError(f"Invalid excitation type. Supported types: ``acc``, ``vel`` and ``disp``.")

        if unit == 'g':
            self.unit_scale = 9.81
        elif unit == 'ms2':
            self.unit_scale = 1        
        else:
            raise ValueError("Invalid unit selected. Supported units: 'g' and 'ms2'.")
                

    def set_random_load(self, signal_data=None, T=None, unit='ms2', method='convolution', bins=None):
        """
        Set random signal load parameters

        :param signal_data: tuple containing (time history data, dt) or (psd data, frequency vector)
        :param T: time duration [s]
        :param unit: unit of the signal (supported: 'g' and 'ms2') Parameter only needed for fds calculation
        :param method: method to calculate ERS and FDS (supported: 'convolution' and 'psd_averaging'). Only needed for random time signal
        :param bins: number of bins for PSD averaging method. Only neede for psd averaging method
        """

        # Signal data must be a tuple
        if isinstance(signal_data, tuple) and len(signal_data) == 2:
        
        # If input is time signal
            if isinstance(signal_data[0], np.ndarray) and isinstance(signal_data[1], (int, float)):
                self.signal_type = 'random_time'
                self.time_data = signal_data[0]  # time-history
                self.dt = signal_data[1] # Sampling interval

                if method in ['convolution', 'psd_averaging']:
                    self.method = method

                else:
                    raise ValueError('Invalid method. Supported methods: ``convolution`` and ``psd_averaging``')

                if isinstance(bins, int):
                    self.bins = bins
                if isinstance(T, (int, float)):
                    print('Time duration ``T`` is not needed for random time signal')
                self.T = len(self.time_data) * self.dt
        
        # If input is PSD
            elif isinstance(signal_data[0], np.ndarray) and isinstance(signal_data[1], np.ndarray):
                self.signal_type = 'random_psd'
                self.psd_data = signal_data[0]
                self.psd_freq = signal_data[1]
                
                if isinstance(T, (int, float)):
                    self.T = T
                else:
                    raise ValueError('Time duration ``T`` must be provided')

            else:
                raise ValueError('Invalid input. Expected a tuple containing (time history data, fs) or (psd data, frequency vector)')
            

        if unit == 'g':
            self.unit_scale = 9.81
        elif unit == 'ms2':
            self.unit_scale = 1
        else:
            raise ValueError("Invalid unit selected. Supported units: 'g' and 'ms2'.")


    def get_ers(self):
        """
        get extreme response spectrum (ERS) of a signal.

        The unit of the ERS corresponds to the unit of the signal, no scaling is applied.

        """        
        if self.signal_type == 'sine':
            self.ers = self._get_sine_ers_fds(output='ERS')
        
        if self.signal_type == 'sine_sweep':
            self.ers = self._get_sine_sweep_ers_fds(output='ERS')
        
        if self.signal_type == 'random_psd':
            self.ers = self._get_random_psd_ers_fds(output='ERS')
        
        if self.signal_type == 'random_time':
            if self.method == 'convolution':
                self.ers = self._get_random_time_ers_fds(output='ERS')
            elif self.method == 'psd_averaging':
                tools.psd_averaging(self)
                self.ers = self._get_random_psd_ers_fds(output='ERS')
                


    def get_fds(self, k, C=1, p=1):
        """
        get fatigue damage spectrum (FDS) of a signal.

        Material parameters k and C must be provided, as defined in equation:

        ``N * s**k = C``,
        
        where ``N`` is the number of cycles and ``s`` is the stress amplitude.

        Additionally, constant ``p`` (proportionality between peak stress and maximum relative displacement) must be provided, as defined by:

        ``sigma_p = p * z_p``

        Correct unit must be selected in `set_random_load` method. If unit is ``g``, signal is scaled to ``m/s^2`` before FDS calculation, because the FDS theory is based on SI base units.

        NOTE:
        Naming of material parameters slightly differs from the notation in literature by Lalanne [1] (``b,C,K`` -> ``k,C,p``). This is done due to the consistency with the established package in this ecosystem (FLife <https://github.com/ladisk/FLife>).

        Alternative material parameters
        -------------------------------
        If you have parameters ``b`` and ``sigma_f`` from equation 
        ``sigma_a = sigma_f * (2*N)**b`` you can convert them to ``k`` and ``C`` using 
        the `tools.material_parameters_convert` function.

        References
        ----------
        1. C. Lalanne, Mechanical Vibration and Shock: Specification development, London, England: ISTE Ltd and John Wiley & Sons, 2009

        :param k: S-N curve slope from Basquin equation
        :param C: material constant from Basquin equation (default: C=1)
        :param p: constant of proportionality between stress and deformation (default: p=1)
        """
        
        if all(isinstance(attr, (int, float)) for attr in [k, C, p]):
            self.k = k
            self.C = C
            self.p = p
        else:
            raise ValueError('Material parameters: k, C and p must be provided')

        if self.signal_type == 'sine':
            self.fds = self._get_sine_ers_fds(output='FDS')

        if self.signal_type == 'sine_sweep':
            self.fds = self._get_sine_sweep_ers_fds(output='FDS')

        if self.signal_type == 'random_psd':
            self.fds = self._get_random_psd_ers_fds(output='FDS')

        if self.signal_type == 'random_time':
            if self.method == 'convolution':
                self.fds = self._get_random_time_ers_fds(output='FDS')   
            elif self.method == 'psd_averaging':
                tools.psd_averaging(self)
                self.fds = self._get_random_psd_ers_fds(output='FDS')


    def plot_ers(self, new_figure=True, grid=True, *args, **kwargs):
        """
        Plot the extreme response spectrum (ERS) of the signal

        :param new_figure: create a new figure. Choose False for adding plot to existing Figure (default: True)
        :param grid: show grid (default: True)	

        """
        if hasattr(self, 'ers'):
            if new_figure:
                plt.figure()
            plt.plot(self.f0_range, self.ers, *args, **kwargs)
            plt.xlabel('Frequency [Hz]')
            if self.unit_scale == 9.81:
                plt.ylabel(f'ERS [g]')
            elif self.unit_scale == 1:
                plt.ylabel(f'ERS [m/s²]')
            plt.title('Extreme Response Spectrum')
            # check if there are is label in kwargs and add legend
            if 'label' in kwargs:
                plt.legend()

            if grid:
                plt.grid(visible=True)
            else:
                plt.grid(visible=False)
        else:
            raise ValueError('ERS not calculated. Run get_ers method first')         


    def plot_fds(self, new_figure=True, grid=True, *args, **kwargs):
        """
        Plot the fatigue damage spectrum (FDS) of the signal
        """
        if hasattr(self, 'fds'):
            if new_figure:
                plt.figure()
            plt.semilogy(self.f0_range, self.fds, *args, **kwargs)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('FDS [Damage]')
            if 'label' in kwargs:
                plt.legend()
            plt.title('Fatigue Damage Spectrum')    
            if grid:
                plt.grid(visible=True)          
            else:
                plt.grid(visible=False)
        else:  
            raise ValueError('FDS not calculated. Run get_fds method first')
        

    def _get_sine_ers_fds(self, output=None):
        """
        Internal function for calculating ERS and FDS of a sine signal.
        """

        omega_0i = 2 * np.pi * self.f0_range

        # Getting the ERS with self.get_ers()
        if output == 'ERS':

            R_i = -self.amp * (omega_0i)**self.a / (np.sqrt((1 - (self.sine_freq / self.f0_range)**2)**2 + (self.sine_freq / (self.Q * self.f0_range))**2))
            return np.abs(R_i) 

        # Getting the FDS with self.get_fds()
        elif output == 'FDS':

            if not hasattr(self, 't_total'):
                raise ValueError('Missing parameter `t_total`.')

            h = self.sine_freq / self.f0_range
            D_i = self.p**self.k / self.C * self.f0_range * self.t_total * self.amp**self.k * omega_0i**(self.k * (self.a - 2)) * h**(self.a * self.k + 1) / ((1 - h**2)**2 + (h / self.Q)**2)**(self.k / 2)
            return D_i
        
    def _get_sine_sweep_ers_fds(self, output=None):
        """
        Internal function for calculating ERS and FDS of a sine sweep signal.
        """
        
        R_i_all = np.zeros((len(self.f0_range), len(self.const_amp)))
        fds = np.zeros(len(self.f0_range))
        ers = np.zeros(len(self.f0_range))
        
        for i in range(len(self.f0_range)):
            omega_0i = 2 * np.pi * self.f0_range[i]

            for n in range(len(self.const_amp)):
                amp = self.const_amp[n]
                f1 = self.const_f_range[n]
                f2 = self.const_f_range[n + 1]
                h1 = f1 / self.f0_range[i]
                h2 = f2 / self.f0_range[i]

                if output == 'FDS':
                    if self.sweep_type is None:
                        raise ValueError("You need to provide either ['linear','lin'] or ['logarithmic','log'] sweep_type.")
                    elif self.sweep_type in ['lin', 'linear']:
                        tb = (self.const_f_range[-1] - self.const_f_range[0]) / self.sweep_rate * 60  # sinusoidal sweep time [s] -> from [Hz/min]
                        dh = (f2 - f1) * self.dt / (self.f0_range[i] * tb)
                        h = np.arange(h1, h2, dh)
                        M_h = h**2 / (h2 - h1)
                    elif self.sweep_type in ['log', 'logarithmic']:
                        tb = 60 * np.log(self.const_f_range[-1] / self.const_f_range[0]) / (self.sweep_rate * np.log(2))  # logarithmic sweep time [s] -> from [oct./min]
                        t = np.arange(0, tb, self.dt)
                        T1 = tb / np.log(h2 / h1)
                        f_t = f1 * np.exp(t / T1)
                        dh = f1 / (T1 * self.f0_range[i]) * np.exp(t / T1) * self.dt
                        h = f_t / self.f0_range[i]
                        M_h = h / (np.log(h2 / h1))
                    else:
                        raise ValueError(f"Invalid method `method`='{self.sweep_type}'. Supported sweep types: 'lin' and 'log'.")
                
                    const = self.p**self.k / self.C * self.f0_range[i] * tb * amp**self.k * omega_0i**(self.k * (self.a - 2))
                    integral = scipy.integrate.trapezoid(M_h * h**(self.a * self.k - 1) / ((1 - h**2)**2 + (h / self.Q)**2)**(self.k / 2), x=h)
                    fds[i] += const * integral

                elif output == 'ERS':
                    if self.f0_range[i] <= f1:
                        Omega_1 = 2 * np.pi * f1
                        R_i = Omega_1**self.a * amp / (np.sqrt((1 - h1**2)**2 + (h1 / self.Q)**2))  # page 32/501 eq. [1.22]
                    elif self.f0_range[i] >= f2:
                        Omega_2 = 2 * np.pi * f2
                        R_i = Omega_2**self.a * amp / (np.sqrt((1 - h2**2)**2 + (h2 / self.Q)**2))  # page 32/501 eq. [1.23]
                    else:
                        R_i = omega_0i**self.a * amp * self.Q  # page 31/501 eq. [1.21] 
                    R_i_all[i, n] = R_i

            ers[i] = max(R_i_all[i, :])
        
        if output == 'ERS':
            return ers
        elif output == 'FDS':
            return fds

    def _get_random_psd_ers_fds(self, output=None):
        """
        Internal function for calculating ERS and FDS of a random signal in frequency domain.
        """
        
        fds = np.zeros(len(self.f0_range))
        ers = np.zeros(len(self.f0_range))     
        
        # constants
        C0 = np.pi / (4 * self.damp)
        C_disp = C0 * 1 / ((2 * np.pi)**4 * self.f0_range**3)
        C_vel = C0 * 1 / ((2 * np.pi)**2 * self.f0_range)
        C_acc = C0 * self.f0_range

        # rms sums
        z_rms_2 = tools.rms_sum(f_0=self.f0_range, psd_freq=self.psd_freq, psd_data=self.psd_data, damp=self.damp, motion='rel_disp') * C_disp
        z_rms = np.sqrt(z_rms_2)
        
        dz_rms_2 = tools.rms_sum(f_0=self.f0_range, psd_freq=self.psd_freq, psd_data=self.psd_data, damp=self.damp, motion='rel_vel') * C_vel
        dz_rms = np.sqrt(dz_rms_2)
        
        if output == 'FDS':  # ddz only needed for FDS calculation
            ddz_rms_2 = tools.rms_sum(f_0=self.f0_range, psd_freq=self.psd_freq, psd_data=self.psd_data, damp=self.damp, motion='rel_acc') * C_acc 
            ddz_rms = np.sqrt(np.abs(ddz_rms_2)) * self.unit_scale

        # ERS calculation
        if output == 'ERS':
            n0 = 1 / np.pi * dz_rms / z_rms
            ers = (2 * np.pi * self.f0_range)**2 * z_rms * np.sqrt(2 * np.log(n0 * self.T))
            return ers
        
        # FDS calculation (damage according to Vol. 0, page 89/198, equation (A1-93))
        elif output == 'FDS':
            z_rms *= self.unit_scale
            dz_rms *= self.unit_scale
            n0 = 1 / np.pi * dz_rms / z_rms
            fds = self.p**self.k / self.C * n0 * self.T * (z_rms * np.sqrt(2))**self.k * gamma(1 + self.k / 2)
            return fds
        
    def _get_random_time_ers_fds(self, output=None):
        """
        Internal function for calculating ERS and FDS of a sine random signal in time domain.
        """

        if output == 'ERS':
            ers = np.zeros(len(self.f0_range))
            for i in tqdm(range(len(self.f0_range))):               
                z = tools.response_relative_displacement(self.time_data, self.dt, f_0=self.f0_range[i], damp=self.damp)
                R_i = np.max(z) * (2 * np.pi * self.f0_range[i])**2 
                ers[i] = R_i
            return ers
        
        if output == 'FDS':
            fds = np.zeros(len(self.f0_range))
            
            for i in tqdm(range(len(self.f0_range))):                    
                z = tools.response_relative_displacement(self.time_data * self.unit_scale, self.dt, f_0=self.f0_range[i], damp=self.damp)
                
                rf = rainflow.count_cycles(z)
                rf = np.asarray(rf)
                cyc_sum = np.sum(rf[:,1] * 2 * (rf[:,0] / 2)**self.k)  # *2 and /2 because rainflow returns cycles and ranges, fds theory is defined for half cycles and amplitudes
                D_i = self.p**self.k / (self.C) * cyc_sum
                fds[i] = D_i
            return fds


        
