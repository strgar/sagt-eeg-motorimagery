import numpy as np
from graph.plv import compute_plv
from graph.gaussian_kernel import gaussian_kernel




def build_adaptive_adjacency(eeg_signal: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
"""
Adaptive adjacency construction
A = alpha * A_PLV + beta * A_Gaussian
"""
A_plv = compute_plv(eeg_signal)
A_gauss = gaussian_kernel(A_plv)
return alpha * A_plv + beta * A_gauss




# =============================================================
from graph.plv import compute_plv
from graph.gaussian_kernel import gaussian_kernel




def build_plv_gaussian_adjacency(eeg_signal: np.ndarray) -> np.ndarray:
plv = compute_plv(eeg_signal)
return gaussian_kernel(plv)
