#!/bin/bash
# Compile ChurnFlux conference paper
# Requires: pdflatex + bibtex (or xelatex + biber)

echo "=== Compiling ChurnFlux Paper ==="

# Clean auxiliary files
rm -f churnflux_paper.aux churnflux_paper.log churnflux_paper.out churnflux_paper.bbl churnflux_paper.blg churnflux_paper.pdf

# First LaTeX pass (generates .aux)
echo "[1/4] LaTeX pass 1..."
pdflatex -interaction=nonstopmode churnflux_paper.tex > /dev/null 2>&1

# BibTeX pass (generates bibliography)
if [ -f churnflux_paper.aux ]; then
  echo "[2/4] BibTeX..."
  bibtex churnflux_paper > /dev/null 2>&1
else
  echo "[ERROR] No .aux file found. Check for LaTeX errors."
  exit 1
fi

# Second LaTeX pass (resolves citations)
echo "[3/4] LaTeX pass 2..."
pdflatex -interaction=nonstopmode churnflux_paper.tex > /dev/null 2>&1

# Third LaTeX pass (resolves cross-references)
echo "[4/4] LaTeX pass 3..."
pdflatex -interaction=nonstopmode churnflux_paper.tex > /dev/null 2>&1

if [ -f churnflux_paper.pdf ]; then
  echo ""
  echo "SUCCESS! PDF generated: churnflux_paper.pdf"
  echo "Pages: $(pdfinfo churnflux_paper.pdf 2>/dev/null | grep Pages || echo 'check manually')"
else
  echo ""
  echo "WARNING: PDF not generated. Check churnflux_paper.log for errors."
  exit 1
fi