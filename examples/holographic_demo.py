import numpy as np
import pandas as pd

from quickinsights import (
    embed_3d_projection,
    plotly_embed_3d,
    volumetric_density_plot,
)


def main():
    X = pd.DataFrame(np.random.randn(2000, 6))
    emb = embed_3d_projection(X)
    print("Embedding shape:", emb["embedding"].shape)

    # Plotly figure (only constructed; not shown in script)
    fig_res = plotly_embed_3d(emb["embedding"], size=2)
    print("Plotly success:", fig_res.get("success", False))

    vol = volumetric_density_plot(X, bins=16)
    print("Volume shape:", vol["volume"].shape)


if __name__ == "__main__":
    main()


