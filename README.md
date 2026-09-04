# ChurnFlux: Measuring and Recovering Per-Query Tail Recall Under Dynamic Updates in Graph-Based ANN Search

**Author:** Daksh Agarwal (VIT Chennai, India)

## Abstract

Graph-based Approximate Nearest Neighbor Search (ANNS) indexes power retrieval-augmented generation (RAG), recommendation, and agentic-memory systems. In production, these indexes face continuous insert/delete churn that degrades search quality unevenly: average recall stays deceptively stable while a subset of queries catastrophically fail.

We present a systematic study of per-query recall under churn across four datasets (SIFT-128, Fashion-MNIST, GloVe-200, SIFT-1M) and two index families (HNSW, IVF-PQ), reporting means with 95% confidence intervals over independent seeds.

## Key Contributions

1. **Measurement study** showing HNSW average recall degrades under 1% at 40% churn at moderate scale, yet 0.9-1.1% of queries fail at 10K vectors and 13-18% at 50K, invisible to average metrics.

2. **AUC-vs-churn trajectory** for a flux-based instability predictor (AUC = 0.939 [0.927, 0.951] across 5-50% churn), with a Churn Amplification Theorem proving the mean-tail gap grows as Theta(sqrt(p)).

3. **Adaptive per-query effort allocation** that eliminates all failures on SIFT-128 at matched cost, improves p1 tail recall by 39.8% at 200K scale, and raises recall@10 from 0.490 to 0.514 in a retrieval pipeline.

4. **Explanation of IVF-PQ's counterintuitive recall increase** under churn via codebook rebalancing.

## Repository Structure

```
├── src/                        # Source code for experiments
│   ├── run_journal_experiments.py
│   ├── run_journal_full.py
│   ├── analyze_results.py
│   ├── compute_honest.py
│   ├── generate_journal_figures.py
│   ├── probe_glove.py
│   └── probe_timing.py
├── data/                       # Experimental results and logs
├── figures/                    # Generated figures
├── comprehensive_results/      # Comprehensive experimental results
├── experiment_results/         # Experiment outputs
├── jsa_experiment_results/     # JSA experiment outputs
├── ieee_resubmit/              # Paper source (LaTeX)
├── experiments_v2.py           # Main experiment script
├── generate_journal_figures.py # Figure generation
├── generate_graphical_abstract.py
├── compile_all.py              # Compilation script
├── compile_paper.sh
├── references.bib              # Bibliography
└── download_datasets.sh        # Script to download benchmark datasets
```

## Datasets

Benchmark datasets are not included in this repository due to size (~2GB total).
Download them using:

```bash
bash download_datasets.sh
```

Or manually from [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks):
- `sift-128-euclidean.hdf5`
- `fashion-mnist-784-euclidean.hdf5`
- `glove-200-angular.hdf5`
- `sift-1m-euclidean.hdf5`

## Requirements

- Python 3.10+
- hnswlib >= 0.8
- faiss-cpu >= 1.14 (or faiss-gpu)
- numpy >= 2.5
- scipy >= 1.18
- scikit-learn >= 1.9

## Usage

```bash
# Install dependencies
pip install hnswlib faiss-cpu numpy scipy scikit-learn

# Download datasets
bash download_datasets.sh

# Run main experiments
python experiments_v2.py

# Run journal experiments
python src/run_journal_full.py

# Generate figures
python generate_journal_figures.py
```

## Reproducibility

All experiments use fixed random seeds (42-51 for AUC trajectory and RAG evaluations; 41-43 for cross-dataset study). Results are reported as means with 95% confidence intervals over independent seeds.

## Citation

```bibtex
@article{agarwal2026churnflux,
  title={ChurnFlux: Measuring and Recovering Per-Query Tail Recall Under Dynamic Updates in Graph-Based ANN Search},
  author={Agarwal, Daksh},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

- Email: daksh.agarwal2025@vitstudent.ac.in
- GitHub: [@daksh1403](https://github.com/daksh1403)
