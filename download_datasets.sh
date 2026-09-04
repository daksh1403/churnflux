#!/bin/bash
# Download benchmark datasets for ChurnFlux experiments
# Source: ANN-Benchmarks (https://github.com/erikbern/ann-benchmarks)

set -e

echo "Downloading ChurnFlux benchmark datasets..."
echo "Total size: ~2GB"
echo ""

# Create datasets directory
mkdir -p datasets

# Download from ANN-Benchmarks
declare -A FILES=(
    ["sift-128-euclidean.hdf5"]="https://ann-benchmarks.com/sift-128-euclidean.hdf5"
    ["fashion-mnist-784-euclidean.hdf5"]="https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
    ["glove-200-angular.hdf5"]="https://ann-benchmarks.com/glove-200-angular.hdf5"
    ["sift-1m-euclidean.hdf5"]="https://ann-benchmarks.com/sift-1m-euclidean.hdf5"
)

SUCCESS=0
FAILED=0

for filename in "${!FILES[@]}"; do
    url="${FILES[$filename]}"
    echo "Downloading $filename..."
    if curl -L --progress-bar -o "datasets/$filename" "$url" 2>/dev/null; then
        echo "  Done."
        ((SUCCESS++))
    else
        echo "  FAILED. Please download manually from: $url"
        ((FAILED++))
    fi
done

echo ""
echo "========================================="
echo "Downloaded: $SUCCESS files"
echo "Failed: $FAILED files"
echo "========================================="
echo ""
echo "Datasets saved to ./datasets/"
echo ""
echo "Alternative: Download from https://github.com/erikbern/ann-benchmarks"
