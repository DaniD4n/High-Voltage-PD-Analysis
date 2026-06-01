import numpy as np

from src.io import load_data
from src.models import kde_2d, gmm_2d, phase_kde
from src.filters import combined_score, density_filter, find_phase_peaks, probability
from src.visualization import plot_raw, plot_density, plot_structure, plot_filter


PATH = "data/example.parquet"

w, A, A_norm = load_data(PATH)

X = np.column_stack([w, A_norm])

x_grid, density = phase_kde(w)

peaks = find_phase_peaks(density)

n_comp = min(len(peaks), 8)

kde_score = kde_2d(X)
gmm_score = gmm_2d(X, n_components=n_comp)

score = combined_score(kde_score, gmm_score)

mask = density_filter(score, percentile=10)

w_f = w[mask]
A_f = A_norm[mask]

plot_raw(w, A_norm)
plot_density(w, A_norm)
plot_structure(x_grid, density, peaks)
plot_filter(w, A_norm, w_f, A_f)

print(probability(
    x_grid,
    density,
    float(input("Min: ")),
    float(input("Max: "))
))

import matplotlib.pyplot as plt
plt.show()
