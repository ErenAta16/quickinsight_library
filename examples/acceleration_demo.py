import numpy as np

from quickinsights import (
    gpu_available,
    get_array_backend,
    gpu_corrcoef,
    memmap_array,
    chunked_apply,
    benchmark_backend,
)


def main():
    print("GPU usable:", gpu_available())
    X = np.random.randn(3000, 10)
    C = gpu_corrcoef(X)
    print("Corr shape:", C.shape)

    mmap = memmap_array('./quickinsights_output/mmap_example.dat', 'float32', (1000, 8))
    mmap[:] = np.random.randn(1000, 8)
    mmap.flush()
    print("Memmap first row sum:", float(mmap[0].sum()))

    parts = chunked_apply(lambda a: float(np.sum(a)), X, chunk_rows=500)
    print("Chunk results count:", len(parts))

    bench = benchmark_backend(lambda xp: xp.random.randn(400, 400) @ xp.random.randn(400, 400), repeats=1)
    print("Timings:", bench["timings"])


if __name__ == "__main__":
    main()


