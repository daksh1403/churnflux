# ChurnFlux — IEEE Access Resubmission Package

## Contents

| File | Purpose |
|------|---------|
| `churn.tex` | Main manuscript (IEEE Access `ieeeaccess` class, two-column) |
| `ieeeaccess.cls`, `IEEEtran.cls` | IEEE Access LaTeX class files |
| `bullet.png`, `logo.png`, `notaglinelogo.png` | Class-required graphics |
| `churnflux_architecture.png` | Architecture figure |
| `fig_auc.png`, `fig_p1.png`, `fig_rag.png` | **Result figures (publication quality)** |
| `Daksh.jpeg`, `Sobitha.jpg` | Author photos |
| `cover_letter.txt` | Cover letter with AI + prior-draft disclosure |
| `run_full_experiments.py` | Full experiment harness (HNSW + IVF-PQ + RAG) |
| `run_ivfpq_adaptive.py` | IVF-PQ adaptive-transfer experiment (negative result) |
| `gen_tables.py`, `gen_figures.py` | Regenerate tables/figures from results |
| `paper_results.json` | **All** aggregated results with 95% CIs |
| `ivfpq_adaptive_results.json` | IVF-PQ adaptive-transfer results |
| `hnsw_run.log`, `ivfpq_run.log`, `ivfpq_adaptive.log`, `fmnist_10.log` | Raw run logs |
| `_ckpt_hnsw_*.json`, `fmnist_hnsw_10seeds.json`, `ivfpq_results.json` | Checkpoint / raw data |

## Experiments

### HNSW (primary, in `paper_results.json`)
- 3 datasets: SIFT-1M (1,000,000 vec), GloVe-200 (1,180,000), Fashion-MNIST (60,000)
- Churn 5/10/20/30/40/50%, independent index per (seed, rate)
- Seeds: 10 (SIFT-1M, Fashion-MNIST), 5 (GloVe-200); 95% CIs
- Per (seed, rate): avg Recall@10, p1, p5, failure count, AUC of instability predictor,
  ChurnFlux adaptive (top-20% unstable → ef×2), uniform at matched average ef

### IVF-PQ (measurement, in `paper_results.json`)
- Same churn protocol, 5 seeds: recall + quantization error (new vs old vectors)

### IVF-PQ adaptive transfer (negative result, `ivfpq_adaptive_results.json`)
- Probe-and-boost via nprobe (10→20 probe, 40 boost) vs matched uniform nprobe
- **Result: adaptive p1 = 0.000, worse than baseline and uniform** — the ef-defined
  answer-set-flux signal does not transfer to nprobe space. Reported honestly in the
  manuscript as a scoped negative result with mechanism; PQ-aware signal = future work.

### RAG downstream (10 seeds, in `paper_results.json`)
- ChurnFlux adaptive 0.514 vs uniform 0.490 recall@10; p1 0.180 vs 0.140

## Reproducing

```bash
# Requires: python3, hnswlib, faiss, h5py, scipy, scikit-learn, matplotlib
# Datasets (ANN-Benchmarks hdf5) in ../ : sift-1m-euclidean.hdf5, glove-200-angular.hdf5,
# fashion-mnist-784-euclidean.hdf5

.venv/bin/python run_full_experiments.py          # ~3.5 h on Apple M5 (HNSW + IVF + RAG)
.venv/bin/python run_ivfpq_adaptive.py            # IVF-PQ adaptive (negative result)
.venv/bin/python gen_tables.py > generated_tables.tex
.venv/bin/python gen_figures.py                    # fig_auc.png, fig_p1.png, fig_rag.png
```

Compile with **pdfLaTeX** (the `ieeeaccess` class requires pdfTeX; tectonic/XeTeX fails on
`spotcolor.sty` — template limitation, not a manuscript error):

```bash
pdflatex churn.tex && pdflatex churn.tex
```

## Key numbers (95% CI)

- **SIFT-1M @ 20% churn**: avg Recall@10 0.918; p1 0.389; 61.3 failures.
  ChurnFlux p1 0.590 vs uniform p1 0.469 at matched cost.
- **AUC** of instability predictor: SIFT-1M 0.90–0.92, GloVe-200 0.83–0.94, Fashion-MNIST 0.87–1.00.
- **GloVe-200**: p1 = 0.000 at all churn rates (extreme tail collapse; honest limitation).
- **IVF-PQ adaptive**: negative transfer (p1 = 0.000) — reported as a scoped negative result.
- **RAG downstream**: recall@10 0.514 vs 0.490; p1 0.180 vs 0.140.
