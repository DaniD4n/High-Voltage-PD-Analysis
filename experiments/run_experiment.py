import numpy as np
import matplotlib.pyplot as plt
from src.io import load_data
from src.models import kde_2d, gmm_2d, phase_kde
from src.filters import combined_score, density_filter, find_phase_peaks, probability
from src.visualization import (plot_raw, plot_density, plot_structure, plot_filter)


# CONFIG (edit ONLY this for experiments)
CONFIG = {"path": "C:\\Users\\Daniiiii\\Downloads\\Parquets 1\\Corona\\grounded plane 125_Ch1.parquet",
    "kde_bandwidth": 2.5,
    "gmm_max_components": 8,
    "filter_percentile": 10}


# EXPERIMENT CORE----------------------------------------------------------------------------------------------
def run_experiment(cfg):

    # Load data--------------
    w, A, A_norm = load_data(cfg["path"])
    X = np.column_stack([w, A_norm])

    # Phase KDE--------------
    x_grid, density = phase_kde(
        w,
        bandwidth=cfg["kde_bandwidth"]
    )

    # Peak detection---------------
    peaks = find_phase_peaks(density)
    n_comp = min(len(peaks), cfg["gmm_max_components"])

    # Models---------------
    kde_score = kde_2d(X)
    gmm_score = gmm_2d(X, n_components=n_comp)

    # Combine scores-------------------
    score = combined_score(kde_score, gmm_score)

    # Filter---------------------
    mask = density_filter(score, percentile=cfg["filter_percentile"])

    w_f = w[mask]
    A_f = A_norm[mask]

    # Plots--------------------------
    plot_raw(w, A_norm)
    plot_density(w, A_norm)
    plot_structure(x_grid, density, peaks)
    plot_filter(w, A_norm, w_f, A_f)
    plt.show()


    # Output metric-------------------
    result = probability(x_grid, density, float(input("Min: ")), float(input("Max: ")))

    print("\nProbability:", result)



# ENTRY POINT------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment(CONFIG)
