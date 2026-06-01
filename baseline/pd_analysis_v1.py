import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks



# DATA--------------------------------------------------


def load_data(path):

    df = pd.read_parquet(path)

    w = df["fase"].to_numpy()
    A = df["amplitude"].to_numpy()

    A_norm = A / (np.max(np.abs(A)) + 1e-12) # so para evitar errors

    return w, A, A_norm



# MODELS----------------------------------------------


def kde_2d(X, bandwidth=0.6):

    model = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth
    )

    model.fit(X)

    score = np.exp(model.score_samples(X))
    score /= np.max(score) + 1e-12

    return score


def gmm_2d(X, n_components=4):

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )

    model.fit(X)

    score = np.exp(model.score_samples(X))
    score /= np.max(score) + 1e-12

    return score


def phase_kde(w,
              bandwidth=2.5,
              points=400):

    x_grid = np.linspace(0, 360, points)

    model = KernelDensity(
        kernel="gaussian",
        bandwidth=bandwidth
    )

    model.fit(w.reshape(-1, 1))

    density = np.exp(
        model.score_samples(
            x_grid.reshape(-1, 1)
        )
    )

    density /= np.max(density) + 1e-12

    return x_grid, density


# FILTERS----------------------------------------------------------------------------

def combined_score(kde_score,
                   gmm_score):

    return kde_score * gmm_score


def density_filter(score,
                   percentile=10):

    return score > np.percentile(
        score,
        percentile
    )



# Analisys------------------------------------------------------------------------------

def find_phase_peaks(density):

    peaks, _ = find_peaks(density)

    return peaks



# PROBABILITY--------------------------------------------------------------------

def probability(x, y, a, b):

    mask = (x >= a) & (x <= b)

    return np.trapezoid(
        y[mask],
        x[mask]
    )



# PLOTS---------------------------------------------

def plot_raw(w, A_norm):

    plt.figure()
    plt.scatter(w, A_norm, s=2, alpha=0.3)
    plt.title("Raw PRPD")
    plt.xlim(0, 360)
    plt.ylim(-1, 1)


def plot_density(w,A_norm):

    plt.figure()
    plt.hist2d(w,A_norm,bins=200,cmap="inferno")
    plt.colorbar()
    plt.title("Density")


def plot_structure(x_grid,density,peaks):

    plt.figure()
    plt.plot(x_grid,density,label="KDE")

    plt.scatter(x_grid[peaks],density[peaks],color="red")

    plt.xlim(0, 360)
    plt.legend()


def plot_filter(w, A_norm, w_f, A_f):

    plt.figure()
    plt.scatter(w,A_norm,s=2,alpha=0.2,label="raw")
    plt.scatter(w_f, A_f, s=2, label="filtered")
    plt.legend()
    plt.title("2D KDE + GMM Filter")



# Principal (main)----------------------------------------------------------------------

PATH = (r"C:\Users\Daniiiii\Downloads\Parquets 1\Internal\in oil_125MSa_Ch1.parquet")

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

plot_raw(w,A_norm)

plot_density( w, A_norm)

plot_structure( x_grid, density, peaks)

plot_filter( w, A_norm, w_f, A_f)

plt.show()

print(probability(x_grid, density, float(input("Mínimo: ")), float(input("Máximo: "))))
