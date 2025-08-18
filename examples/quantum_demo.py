import numpy as np
import pandas as pd

from quickinsights import (
    quantum_superposition_sample,
    amplitude_pca,
    quantum_correlation_map,
)


def main():
    X = pd.DataFrame(np.random.randn(10000, 12))
    samp = quantum_superposition_sample(X, n_samples=1000)
    print("Sample indices (first 10):", samp["indices"][:10])

    pca = amplitude_pca(X, n_components=5)
    print("PCA components shape:", pca["components"].shape)

    qc = quantum_correlation_map(X, n_blocks=3, block_size=4000)
    print("Correlation shape:", qc["correlation"].shape)


if __name__ == "__main__":
    main()


