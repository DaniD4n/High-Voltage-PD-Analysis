import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def kde_2d(X, bandwidth=0.6):
    model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    model.fit(X)

    score = np.exp(model.score_samples(X))
    return score / (np.max(score) + 1e-12)


def gmm_2d(X, n_components=4):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    )

    model.fit(X)

    score = np.exp(model.score_samples(X))
    return score / (np.max(score) + 1e-12)


def phase_kde(w, bandwidth=2.5, points=400): #1D KDE for plotting and probability 
    import numpy as np
    from sklearn.neighbors import KernelDensity

    x_grid = np.linspace(0, 360, points)

    model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    model.fit(w.reshape(-1, 1))

    density = np.exp(model.score_samples(x_grid.reshape(-1, 1)))
    return x_grid, density / (np.max(density) + 1e-12)
