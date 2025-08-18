"""
QuickInsights - Comprehensive End-to-End Demo Script
Covers: Pandas/Numpy integration, ML pipeline, Feature/Model selection, Dask big data,
Neural patterns, Quantum-inspired analytics, Holographic 3D (non-VR), Acceleration (GPU/Memory).
"""

import os
import json
import time
import numpy as np
import pandas as pd

import quickinsights as qi

OUTPUT_DIR = "./quickinsights_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title: str):
	print(f"\n=== {title} ===")


def save_json(obj, path):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=2, default=str)
	print(f"Saved: {path}")


def save_df(df: pd.DataFrame, path: str, index: bool = False):
	df.to_csv(path, index=index)
	print(f"Saved: {path}")


def main():
	section("Utility & System Status")
	try:
		status = qi.get_utility_status()
		print(status)
		qi.print_utility_status()
	except Exception as e:
		print("Utility status unavailable:", e)

	# ------------------------------------------------------------
	# Synthetic Data
	# ------------------------------------------------------------
	section("Create Synthetic Data")
	rng = np.random.default_rng(42)
	n_rows = 5000
	df = pd.DataFrame(
		{
			"num_0": rng.normal(size=n_rows),
			"num_1": rng.normal(loc=2.0, scale=3.0, size=n_rows),
			"num_2": rng.normal(loc=-1.0, scale=0.5, size=n_rows),
			"cat_0": rng.choice(["A", "B", "C"], size=n_rows, p=[0.5, 0.3, 0.2]),
			"cat_1": rng.choice(["X", "Y"], size=n_rows),
		}
	)
	# Secondary DataFrame for merge demo
	df2 = df.copy()
	df2["join_key"] = np.arange(n_rows)
	df["join_key"] = np.arange(n_rows)
	print("df shape:", df.shape)

	# ------------------------------------------------------------
	# Pandas Integration
	# ------------------------------------------------------------
	section("Pandas Integration")
	try:
		group_res = qi.smart_group_analysis(
			df,
			group_columns=["cat_0"],
			value_columns=["num_0", "num_1", "num_2"],
			include_visualizations=False,
			save_results=True,
			output_dir=OUTPUT_DIR,
		)
		print("Group analysis keys:", list(group_res.keys()))

		pivot_res = qi.smart_pivot_table(
			df,
			index_columns=["cat_0"],
			columns=["cat_1"],
			values=["num_0", "num_1"],
			aggfunc="mean",
			include_visualizations=False,
			save_results=True,
			output_dir=OUTPUT_DIR,
		)
		print("Pivot done; shape:", pivot_res["pivot_shape"])

		merge_res = qi.intelligent_merge(df, df2, strategy="auto", save_results=True, output_dir=OUTPUT_DIR)
		print("Merge used keys:", merge_res.get("merge_keys"))
	except Exception as e:
		print("Pandas integration error:", e)

	# ------------------------------------------------------------
	# NumPy Integration
	# ------------------------------------------------------------
	section("NumPy Integration")
	try:
		X_np = df.select_dtypes(include=float).to_numpy()
		math_res = qi.auto_math_analysis(X_np, include_visualizations=False, save_results=False)
		print("Math analysis metrics:", list(math_res.keys()))
		save_json(math_res, f"{OUTPUT_DIR}/numpy_math_analysis.json")
	except Exception as e:
		print("NumPy integration error:", e)

	# ------------------------------------------------------------
	# ML: Feature Selection, Model Selection, Auto ML Pipeline
	# ------------------------------------------------------------
	section("ML Pipeline (Feature/Model Selection)")
	try:
		# Build a classification target
		y_clf = (df["num_0"] + 0.5 * df["num_1"] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
		X_ml = df.select_dtypes(include=[float])

		n_feat = min(10, X_ml.shape[1])
		feat_res = qi.smart_feature_selection(X_ml, y_clf, n_features=n_feat)
		print("Selected features:", feat_res.get("consensus_features"))

		model_sel = qi.intelligent_model_selection(X_ml, y_clf, task_type="classification", cv_folds=3)
		print("Best model (classification):", model_sel["best_model"])

		ml_pipe = qi.auto_ml_pipeline(X_ml, y_clf, task_type="classification", save_results=True, output_dir=OUTPUT_DIR)
		print("Auto ML pipeline score:", ml_pipe.get("score"))
	except Exception as e:
		print("ML pipeline error:", e)

	# ------------------------------------------------------------
	# Dask Integration (Big Data)
	# ------------------------------------------------------------
	section("Dask Integration (Big Data)")
	try:
		# Larger data for Dask
		n_big = 100_000
		df_big = pd.DataFrame(
			{
				"num_0": rng.normal(size=n_big),
				"num_1": rng.normal(loc=2.0, scale=3.0, size=n_big),
				"num_2": rng.normal(loc=-1.0, scale=0.5, size=n_big),
				"cat_0": rng.choice(["A", "B", "C"], size=n_big),
			}
		)
		start = time.time()
		dask_res = qi.smart_dask_analysis(df_big, analysis_type="descriptive", n_workers=2, memory_limit="1GB")
		print("Dask analysis keys:", list(dask_res.keys()))
		print("Dask exec time:", dask_res.get("performance", {}).get("execution_time"))
		save_json(dask_res, f"{OUTPUT_DIR}/dask_descriptive.json")

		# Distributed compute demo
		def chunk_sum(arr):
			import numpy as _np
			return float(_np.sum(arr))

		chunks = [rng.normal(size=(1000, 10)) for _ in range(4)]
		dist_res = qi.distributed_compute(chunk_sum, chunks, n_workers=2, memory_limit="1GB", show_progress=False)
		print("Distributed results (sample):", dist_res["results"][:2])

		# Big data pipeline demo
		ops = [
			{"name": "filter_numeric", "type": "filter", "params": {"condition": "num_0 > 0"}},
			{"name": "select_columns", "type": "select", "params": {"columns": ["num_0", "num_1", "num_2", "cat_0"]}},
			{"name": "groupby_analysis", "type": "groupby", "params": {"group_column": "cat_0", "aggregation_column": "num_0", "agg_function": "mean"}},
		]
		pipe_res = qi.big_data_pipeline(df_big, operations=ops, n_workers=2, memory_limit="1GB", save_intermediate=True, output_dir=OUTPUT_DIR)
		print("Pipeline operations:", [p["operation"] for p in pipe_res["pipeline_operations"]])
		print("Final shape:", pipe_res["final_shape"])
		print("Dask section time:", round(time.time() - start, 2), "s")
	except Exception as e:
		print("Dask integration unavailable or failed:", e)

	# ------------------------------------------------------------
	# Neural Patterns (Neural-inspired)
	# ------------------------------------------------------------
	section("Neural Patterns")
	try:
		patterns = qi.neural_pattern_mining(df.select_dtypes(float), n_patterns=5)
		print("Pattern counts:", patterns["counts"])
		save_df(patterns["pattern_centers"], f"{OUTPUT_DIR}/neural_pattern_centers.csv")

		anoms = qi.autoencoder_anomaly_scores(df.select_dtypes(float), hidden_ratio=0.5, max_iter=150)
		print("Anomaly method:", anoms["method"], "| Hidden dim:", anoms["hidden_dim"])
		save_json({"scores_head": anoms["scores"][:20].tolist()}, f"{OUTPUT_DIR}/neural_anomaly_scores_head.json")

		series = pd.Series(np.sin(np.linspace(0, 30, 4000)) + 0.05 * rng.normal(size=4000))
		sigs = qi.sequence_signature_extract(series, window=128, step=64, n_components=3)
		print("Signature windows:", sigs["n_windows"])
		save_df(sigs["signatures"], f"{OUTPUT_DIR}/neural_sequence_signatures.csv")
	except Exception as e:
		print("Neural patterns error:", e)

	# ------------------------------------------------------------
	# Quantum-inspired Analytics
	# ------------------------------------------------------------
	section("Quantum-inspired Analytics")
	try:
		samp = qi.quantum_superposition_sample(df.select_dtypes(float), n_samples=1000)
		print("Superposition sample idx (10):", samp["indices"][:10])
		save_df(samp["subset"], f"{OUTPUT_DIR}/quantum_sample_subset.csv")

		apca = qi.amplitude_pca(df.select_dtypes(float), n_components=5)
		print("Amplitude PCA comps:", apca["components"].shape)
		save_df(apca["components"], f"{OUTPUT_DIR}/quantum_amplitude_pca.csv")

		qcorr = qi.quantum_correlation_map(df.select_dtypes(float), n_blocks=3, block_size=2000)
		print("Quantum corr shape:", qcorr["correlation"].shape)
		save_df(qcorr["correlation"], f"{OUTPUT_DIR}/quantum_correlation_map.csv", index=True)
	except Exception as e:
		print("Quantum analytics error:", e)

	# ------------------------------------------------------------
	# Holographic 3D (non‑VR)
	# ------------------------------------------------------------
	section("Holographic 3D (non-VR)")
	try:
		emb = qi.embed_3d_projection(df.select_dtypes(float), method="pca")
		print("Embedding shape:", emb["embedding"].shape)
		save_df(emb["embedding"], f"{OUTPUT_DIR}/holo_embedding_3d.csv")

		plot_res = qi.plotly_embed_3d(emb["embedding"], size=2)
		print("Plotly success:", plot_res.get("success", False))

		vol = qi.volumetric_density_plot(df.select_dtypes(float), bins=24)
		print("Volume shape:", vol["volume"].shape)
	except Exception as e:
		print("Holographic 3D error:", e)

	# ------------------------------------------------------------
	# Acceleration (GPU/Memory)
	# ------------------------------------------------------------
	section("Acceleration (GPU/Memory)")
	try:
		print("GPU usable:", qi.gpu_available())
		X = df.select_dtypes(float).to_numpy()
		C = qi.gpu_corrcoef(X)
		print("Corr shape:", C.shape)
		mmap = qi.memmap_array(f"{OUTPUT_DIR}/mmap_demo.dat", "float32", (1000, 8))
		mmap[:] = np.random.randn(1000, 8)
		mmap.flush()

		parts = qi.chunked_apply(lambda a: float(np.sum(a)), X, chunk_rows=1000)
		print("Chunked parts:", len(parts))

		bench = qi.benchmark_backend(lambda xp: xp.random.randn(600, 600) @ xp.random.randn(600, 600), repeats=1)
		print("Benchmark timings:", bench["timings"])
	except Exception as e:
		print("Acceleration error:", e)

	section("Done")
	print(f"All results saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()


