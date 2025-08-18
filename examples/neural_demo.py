import numpy as np
import pandas as pd

from quickinsights import (
    neural_pattern_mining,
    autoencoder_anomaly_scores,
    sequence_signature_extract,
)


def main():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(5000, 8)), columns=[f"f{i}" for i in range(8)])

    patterns = neural_pattern_mining(df, n_patterns=5)
    print("Patterns:", patterns["counts"])  # cluster counts

    anoms = autoencoder_anomaly_scores(df)
    print("Anomaly scores (first 10):", anoms["scores"][:10])

    s = pd.Series(np.sin(np.linspace(0, 50, 3000)) + 0.1 * rng.normal(size=3000))
    sigs = sequence_signature_extract(s, window=128, step=32, n_components=3)
    print("Signature windows:", sigs["n_windows"])  # number of windows


if __name__ == "__main__":
    main()


