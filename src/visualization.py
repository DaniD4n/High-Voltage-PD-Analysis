import matplotlib.pyplot as plt
import numpy as np

def plot_sine(max_val):
    x = np.linspace(0, 360, 500)
    y = np.sin(np.deg2rad(x)) * max_val*1.1
    plt.plot(x, y, alpha=0.3, label="Sine", color='red')

def plot_raw(w, A_norm):
    plt.figure()
    plt.scatter(w, A_norm, s=2, alpha=0.3)
    plt.title("Raw PRPD")
    plt.xlim(0, 360)
    plt.ylim(-1, 1)
    plot_sine(max(A_norm))
    plt.show()

def plot_density(w, A_norm):
    plt.figure()
    plt.hist2d(w, A_norm, bins=200, cmap="inferno")
    plt.colorbar()
    plt.title("Density")
    plot_sine(max(A_norm))
    plt.show()

def plot_structure(x_grid, density, peaks):
    plt.figure()
    plt.plot(x_grid, density)
    plt.scatter(x_grid[peaks], density[peaks], color="red")
    plt.xlim(0, 360)
    plot_sine(max(density))
    plt.show()

def plot_filter(w, A_norm, w_f, A_f):
    plt.figure()
    plt.scatter(w, A_norm, s=2, alpha=0.2, label="raw")
    plt.scatter(w_f, A_f, s=2, label="filtered")
    plt.legend()
    plt.title("Filter result")
    plot_sine(max(A_norm))
    plt.show()
