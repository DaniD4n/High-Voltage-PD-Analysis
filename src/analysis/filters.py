import numpy as np
from scipy.signal import find_peaks

def combined_score(kde_score, gmm_score):
    return kde_score * gmm_score


def density_filter(score, percentile=10):
    return score > np.percentile(score, percentile)


def find_phase_peaks(density):
    peaks, _ = find_peaks(density)
    return peaks


def probability(x, y, a, b):
    mask = (x >= a) & (x <= b)
    return np.trapezoid(y[mask], x[mask])
