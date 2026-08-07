# tests/conftest.py
import os
import numpy as np
import pytest

# Optional: real pyeCAP imports (guarded)
try:
    from pyeCAP import Ephys, Stim, ECAP
except Exception:
    Ephys = Stim = ECAP = None

# ----------------------------
# Synthetic (fast) test fixtures
# ----------------------------
@pytest.fixture(scope="session")
def fs():
    return 10_000  # 10 kHz

@pytest.fixture(scope="session")
def t(fs):
    T = 2.0
    n = int(T * fs)
    return np.arange(n) / fs

@pytest.fixture(scope="session")
def synthetic_two_ch_ecap(fs, t):
    """Return (data[2,n], event_times np.ndarray)."""
    def pulse(tt, t0):
        win = (tt >= t0) & (tt < t0 + 0.010)
        x = np.zeros_like(tt)
        x[win] = np.sin(2*np.pi*300*(tt[win]-t0)) * np.exp(-(tt[win]-t0)/0.004)
        return x

    ch0 = np.zeros_like(t)
    ch1 = np.zeros_like(t)
    event_times = np.arange(0.100, 1.000, 0.100)
    for t0 in event_times:
        ch0 += 0.5 * pulse(t, t0)
        ch1 += 0.8 * pulse(t, t0) + 0.02
    data = np.vstack([ch0, ch1])  # (2, n)
    return data, event_times

# ----------------------------
# Optional real-data fixtures (skipped unless configured)
# ----------------------------
@pytest.fixture(scope="session")
def tdt_block_path():
    path = os.getenv("TDT_BLOCK_PATH")  # set this in your env if you want real I/O tests
    if not path or not os.path.isdir(path):
        pytest.skip("No real TDT block configured (set TDT_BLOCK_PATH to enable).")
    return path

@pytest.fixture(scope="session")
def real_ephys(tdt_block_path):
    if Ephys is None:
        pytest.skip("pyeCAP Ephys not importable in this environment.")
    return Ephys(tdt_block_path)

@pytest.fixture(scope="session")
def real_stim(tdt_block_path):
    if Stim is None:
        pytest.skip("pyeCAP Stim not importable in this environment.")
    return Stim(tdt_block_path)

@pytest.fixture(scope="session")
def real_ecap(real_ephys, real_stim):
    if ECAP is None:
        pytest.skip("pyeCAP ECAP not importable in this environment.")
    # Adjust trigger_channel/x_lim to your setup if needed
    return ECAP(real_ephys, real_stim, trigger_channel="Channel 6", x_lim=(0.0, 0.010))
