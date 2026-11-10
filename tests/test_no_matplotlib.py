import json
import os
import subprocess
import sys

import numpy as np

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import FatigueDS

# Runs in a separate interpreter in which matplotlib cannot be imported.
script = """
import json
import sys

sys.modules['matplotlib'] = None  # every import of matplotlib raises ImportError
sys.path.insert(0, sys.argv[1])

import numpy as np
import FatigueDS

sd = FatigueDS.Spectrum(freq_data=(20, 200, 5), damp=0.05)
sd.set_random_load((np.array([1, 1.]), np.array([20, 200.])), unit='ms2', T=3600)
sd.get_ers()
sd.get_fds(k=10, C=1e80, p=6.3e10)

plot_errors = {}
for name in ['plot_ers', 'plot_fds']:
    try:
        getattr(sd, name)()
    except ImportError as e:
        plot_errors[name] = str(e)

print(json.dumps({
    'ers': sd.ers.tolist(),
    'fds': sd.fds.tolist(),
    'plot_errors': plot_errors,
    'matplotlib_imported': any(m.startswith('matplotlib.') for m in sys.modules),
}))
"""


def test_import_and_compute_without_matplotlib():
    """ The package imports and the ERS and FDS are computed without matplotlib; only plotting requires it """
    result = subprocess.run(
        [sys.executable, '-c', script, os.path.join(my_path, '..')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])

    sd = FatigueDS.Spectrum(freq_data=(20, 200, 5), damp=0.05)
    sd.set_random_load((np.array([1, 1.]), np.array([20, 200.])), unit='ms2', T=3600)
    sd.get_ers()
    sd.get_fds(k=10, C=1e80, p=6.3e10)

    assert not out['matplotlib_imported']
    assert np.allclose(out['ers'], sd.ers)
    assert np.allclose(out['fds'], sd.fds)
    for name in ['plot_ers', 'plot_fds']:
        assert f'matplotlib is required for {name}' in out['plot_errors'][name]
