import matplotlib.pyplot as plt
import math

from src.generation.cluster_generator import generate


def run_experiment(n=5000, seed=None):
    if seed is not None:
        import random
        random.seed(seed)

    print(f"Generating {n} synthetic PD points...")

    data = generate(n)

    phase = [d["phase"] * 180 / math.pi for d in data]
    amp = [d["amplitude"] for d in data]

    print(f"Generated: {len(data)} points")

    # ---- PLOT ----
    plt.figure()
    plt.scatter(phase, amp, s=1, alpha=0.5)

    plt.xlim(0, 360)
    plt.ylim(0, 1)

    plt.title("Synthetic PD Scatter (Cluster Generator)")
    plt.xlabel("Phase (degrees)")
    plt.ylabel("Amplitude")

    plt.show()


if __name__ == "__main__":
    run_experiment(n=5000, seed=42)
