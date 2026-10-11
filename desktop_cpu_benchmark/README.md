# Desktop CPU inference benchmark

This folder contains the supplementary code for the CPU-only desktop inference benchmark reported with the study.

## Measurement reported in the manuscript

The measurement used an Intel Core i5-10400 desktop with 16 GB RAM, Windows 11, and TensorFlow 2.15. Inference ran on the CPU. Each prediction used one forward pass on a single 5 x 10 input window with batch size 1. After 100 warm-up runs, 1,000 predictions were measured.

| Measure | Reported value |
| --- | ---: |
| Median inference latency | 6.0 ms |
| 95th-percentile inference latency | 8.2 ms |
| Mean process CPU utilization | 95% |
| Peak process memory (RSS) | 670 MB |

For CPU utilization, 100% represents one fully utilized logical CPU core. Measurements can vary across systems. They describe desktop CPU performance and do not establish performance on a resource-constrained edge device.

## Reproducing the measurement

1. In the trained Kaggle notebook, run `kaggle_export_cell.py` as a new cell after training, when `best_model` and `X_test` are available.
2. Download the four exported files: `bench_weights.npz`, `bench_config.json`, `bench_windows.npy`, and `bench_reference.npy`.
3. Place the four files in this folder's `data/` directory.
4. Install the dependencies with `python -m pip install -r requirements.txt`.
5. Run `python run_benchmark.py` from this folder.

The script writes its reports to `results/`. If the exported files are missing, it runs with random weights and random windows as a dry run. Do not treat dry-run measurements as results for the study.

The exported model and window files are not included in this repository. Generate them from the trained notebook when reproducing the benchmark.
