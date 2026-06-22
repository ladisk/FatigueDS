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
