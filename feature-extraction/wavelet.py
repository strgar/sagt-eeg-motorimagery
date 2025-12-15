import numpy as np
import pywt




def band_energy(signal: np.ndarray) -> float:
"""Compute signal energy."""
return np.sum(signal ** 2)




def extract_wavelet_db5(eeg_signal: np.ndarray, fs: int = 160) -> np.ndarray:
"""
Extract Mu and Beta band energy using DWT (db5).


Parameters
----------
eeg_signal : ndarray (n_channels, n_samples)
Raw EEG signal per trial
fs : int
Sampling frequency (default 160 Hz)


Returns
-------
features : ndarray (n_channels, 2)
[Mu_energy, Beta_energy]
"""
features = []
for ch in range(eeg_signal.shape[0]):
coeffs = pywt.wavedec(eeg_signal[ch], 'db5', level=5)
cA5, cD5, cD4, cD3, cD2, cD1 = coeffs


mu_energy = band_energy(cD3) # 8–13 Hz
beta_energy = band_energy(cD2) # 13–30 Hz


features.append([mu_energy, beta_energy])


return np.array(features)
