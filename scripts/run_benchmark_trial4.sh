#!/usr/bin/env bash
# Run synthony-benchmark for all datasets in trial4
# Initialize conda if available
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate base
fi
# Original data: ./dataset/input_data/
# Synthetic data: ./dataset/synth_data/trial4/
# Output: ./output/benchmark/trial4/

set -e

# Directories
INPUT_DIR="./dataset/input_data"
SYNTH_DIR="./dataset/synth_data/trial4"
OUTPUT_DIR="./output/benchmark/trial4"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Datasets available in trial4
DATASETS=(
    "abalone"
    "Bean"
    "faults"
    "IndianLiverPatient"
    "insurance"
    "Obesity"
    "Shoppers"
    "wilt"
)

# Models to benchmark (based on available synthetic files)
MODELS=(
    "AIM"
    "ARF"
    "AutoDiff"
    "BayesianNetwork"
    "CART"
    "DPCART"
    "Identity"
    "NFlow"
    "SMOTE"
    "TabDDPM"
    "TVAE"
)

# Counter for progress
total=0
success=0
failed=0

echo "=============================================="
echo "Synthony Benchmark Runner - Trial 4"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Synth directory:  $SYNTH_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Loop through datasets and models
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset"
    echo "----------------------------------------------"

    # Check if original file exists
    original_file="$INPUT_DIR/${dataset}.csv"
    if [[ ! -f "$original_file" ]]; then
        echo "  [SKIP] Original file not found: $original_file"
        continue
    fi

    for model in "${MODELS[@]}"; do
        total=$((total + 1))

        # Synthetic file path
        synth_file="$SYNTH_DIR/${dataset}/${dataset}__${model}.csv"

        # Output file path
        output_file="$OUTPUT_DIR/benchmark__${dataset}__${model}.json"

        # Check if synthetic file exists
        if [[ ! -f "$synth_file" ]]; then
            echo "  [SKIP] $model - synthetic file not found"
            continue
        fi

        echo -n "  Benchmarking $model... "

        # Run benchmark using synthony-benchmark
        if synthony-benchmark -r "$original_file" -s "$synth_file" -o "$output_file" > /dev/null 2>&1; then
            echo "[OK]"
            success=$((success + 1))
        else
            echo "[FAILED]"
            failed=$((failed + 1))
        fi
    done

    echo ""
done

echo "=============================================="
echo "Benchmark Complete"
echo "=============================================="
echo "Total:   $total"
echo "Success: $success"
echo "Failed:  $failed"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
