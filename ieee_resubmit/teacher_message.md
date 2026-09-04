Subject: ChurnFlux manuscript — revised and ready for IEEE Access resubmission

Dear Dr. Sobitha Ahila,

I have substantially revised the ChurnFlux manuscript based on our earlier IEEE Access desk rejection, and I believe it is now ready for you to review and resubmit. Below is a summary of what has changed and what I need from you.

## What was done to improve the manuscript

1. **Million-vector-scale experiments.** The previous version was evaluated on small subsets (10K vectors). The revised paper reports full-scale experiments on SIFT-1M (1,000,000 vectors), GloVe-200 (1,180,000), and Fashion-MNIST (60,000) — matching the scale of published ANN papers.

2. **Two index families.** ChurnFlux's adaptive recovery is now demonstrated on both HNSW (via the search effort parameter ef) and IVF-PQ (via nprobe), with matched-cost uniform baselines for a fair comparison.

3. **Ten-seed evaluation with confidence intervals.** All primary results are reported as means over 10 seeds (SIFT-1M, Fashion-MNIST) or 5 seeds (GloVe-200) with 95% confidence intervals — stronger than most published papers in this area, which typically report 1–3 seeds.

4. **Churn sweep 5–50%.** Instead of a single churn rate, the predictor's discriminative power (AUC 0.83–0.94) and the tail-recall recovery are now characterized across six churn intensities.

5. **New results.** On SIFT-1M, ChurnFlux improves the p1 tail recall from 0.44 to 0.60 on HNSW and from 0.14 to 0.24 on IVF-PQ, both at matched average cost. A downstream retrieval evaluation (10 seeds) shows recall@10 improving from 0.490 to 0.514.

6. **Honest handling of limitations.** GloVe-200 exhibits extreme tail collapse (p1 = 0 for all methods) and is reported transparently as a limitation; the paper does not overclaim.

7. **Integrity fixes.** Added the required AI-generated-content disclosure to the Acknowledgments; all references were verified and completed with venues/arXiv IDs; abstract trimmed to 214 words (IEEE requires 150–250); IEEE Access template requirements (history/DOI/corresponding-author blocks, author bios) confirmed.

8. **Full reproducibility package.** All experiment code, raw logs, and results JSON are saved alongside the manuscript.

## What I need from you

1. **Compile the manuscript.** The folder `overleaf_upload/` contains everything needed. Upload all 14 files to Overleaf and compile with the default (pdfLaTeX) compiler. Please confirm it builds cleanly — this is the one step I cannot do from here.

2. **Review the cover letter.** It is updated with the required AI disclosure and prior-draft statement. Please review before submission.

3. **ORCID and manuscript type.** Your account will need a publicly visible ORCID, and we should select "Research Article" as the manuscript type in the submission portal.

4. **Important — please advise on resubmission policy.** The previous submission was desk-rejected with "resubmission not permitted" language in the decision letter. IEEE Access's policy states that manuscripts previously rejected without permission to resubmit may be immediately rejected if resubmitted. Because this is a substantially revised manuscript with new experiments, I would recommend emailing ieeeaccess@ieee.org first to ask whether a heavily revised version may be submitted as a new manuscript. Please advise on how you would like to handle this — it affects whether we should submit to IEEE Access or target a different venue (e.g., Journal of Systems Architecture, for which the manuscript was originally prepared).

## Files

- `overleaf_upload/` — manuscript + class files + figures (upload all to Overleaf)
- `churn.tex` — the manuscript
- `cover_letter.txt` — updated cover letter
- `paper_results.json`, `ivfpq_adaptive_results.json` — all experimental results
- `README.md` — reproducibility instructions

Thank you for your guidance on this. I am happy to make any further revisions you recommend.
