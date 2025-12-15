import numpy as np
from scipy.signal import hilbert




def compute_plv(eeg_signal: np.ndarray) -> np.ndarray:
"""
Compute Phase Locking Value (PLV).


Parameters
----------
eeg_signal : ndarray (n_channels, n_samples)


Returns
-------
plv : ndarray (n_channels, n_channels)
"""
n_ch = eeg_signal.shape[0]
analytic_signal = hilbert(eeg_signal)
phase = np.angle(analytic_signal)


plv = np.zeros((n_ch, n_ch))
for i in range(n_ch):
for j in range(n_ch):
phase_diff = phase[i] - phase[j]
plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
return plv
