import numpy as np

def gaussian_kernel(plv: np.ndarray, sigma: float = 0.5) -> np.ndarray:
"""Apply Gaussian kernel to PLV matrix."""
return np.exp(-((1 - plv) ** 2) / (2 * sigma ** 2))
