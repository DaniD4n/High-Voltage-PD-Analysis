import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_parquet(path)

    w = df["fase"].to_numpy()
    A = df["amplitude"].to_numpy()

    A_norm = A / (np.max(np.abs(A)) + 1e-12)

    return w, A, A_norm
