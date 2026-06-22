"""
A project template for the sdPy effort..
"""

__version__ = "0.3.1"
from .spectrum import Spectrum
from .mission_synthesis import (
    MissionSynthesis,
    combine_fds,
    envelope_ers,
    invert_fds_to_psd,
)
from . import tools
