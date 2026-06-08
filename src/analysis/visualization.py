import matplotlib.pyplot as plt
import numpy as np


def add_sine(ax, x, y_min, y_max):
    sine = np.sin(np.deg2rad(x))
    sine_scaled = (sine + 1) / 2
    sine_scaled = sine_scaled * (y_max - y_min) + y_min
    ax.plot(x, sine_scaled, color='red', alpha=0.3, label="AC reference")


def plot_raw(w, A_norm):
    maximum = max(np.abs(A_norm))
    fig, ax = plt.subplots()
    ax.scatter(w, A_norm, s=2, alpha=0.3)
    ax.set_title("Raw PRPD")
    ax.set_xlim(0, 360)
    ax.set_ylim(-maximum, maximum)

    x = np.linspace(0, 360, 500)
    add_sine(ax, x, -maximum, maximum)


def plot_density(w, A_norm):
    maximum = max(np.abs(A_norm))
    fig, ax = plt.subplots()
    ax.hist2d(w, A_norm, bins=200, cmap="inferno")
    ax.set_title("Density")
    
    x = np.linspace(0, 360, 500)
    add_sine(ax, x, -maximum, maximum)


def plot_structure(x_grid, density, peaks):
    maximum = max(density)
    fig, ax = plt.subplots()
    ax.plot(x_grid, density)
    ax.scatter(x_grid[peaks], density[peaks], color="red")

    ax.set_xlim(0, 360)
    ax.set_ylim(0, maximum * 1.1)

    x = np.linspace(0, 360, 500)
    add_sine(ax, x, 0, maximum * 1.1)


def plot_filter(w, A_norm, w_f, A_f):
    maximum = max(np.abs(A_norm))
    fig, ax = plt.subplots()
    ax.scatter(w, A_norm, s=2, alpha=0.2, label="raw")
    ax.scatter(w_f, A_f, s=2, label="filtered")
    ax.set_title("Filter result")
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_ylim(-maximum, maximum)

    x = np.linspace(0, 360, 500)
    add_sine(ax, x, -maximum, maximum)
