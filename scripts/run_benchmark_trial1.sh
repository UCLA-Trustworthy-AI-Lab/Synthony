#!/usr/bin/env bash
# Run synthony-benchmark for all datasets in trial1
# Initialize conda if available
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate base
fi

set -e

# Directories
INPUT_DIR="./dataset/input_data"
SYNTH_DIR="./dataset/synth_data/trial1"
OUTPUT_DIR="./output/benchmark/trial1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Datasets available in trial1 (note: case-sensitive matching needed)
DATASETS=(
    "abalone"
    "Bean"
    "faults"
    "HTRU2:Htru2"
    "IndianLiverPatient"
    "News"
    "Obesity"
    "Shoppers"
    "Titanic"
    "wilt"
)

# Models to benchmark
MODELS=(
    "AIM"
    "ARF"
    "AutoDiff"
    "BayesianNetwork"
    "CART"
    "DPCART"
    "Identity"
    "PATECTGAN"
    "SMOTE"
    "TabDDPM"
    "TVAE"
)

# Counter for progress
total=0
success=0
failed=0
skipped=0

echo "=============================================="
echo "Synthony Benchmark Runner - Trial 1"
echo "=============================================="
echo "Input directory:  $INPUT_DIR"
echo "Synth directory:  $SYNTH_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Loop through datasets and models
for dataset_entry in "${DATASETS[@]}"; do
    # Handle case where input and synth folder names differ (e.g., HTRU2:Htru2)
    if [[ "$dataset_entry" == *":"* ]]; then
        input_name="${dataset_entry%%:*}"
        synth_folder="${dataset_entry##*:}"
    else
        input_name="$dataset_entry"
        synth_folder="$dataset_entry"
    fi

    echo "Processing dataset: $input_name (folder: $synth_folder)"
    echo "----------------------------------------------"

    # Check if original file exists
    original_file="$INPUT_DIR/${input_name}.csv"
    if [[ ! -f "$original_file" ]]; then
        echo "  [SKIP] Original file not found: $original_file"
        continue
    fi

    for model in "${MODELS[@]}"; do
        total=$((total + 1))

        # Synthetic file path - try both naming conventions
        synth_file="$SYNTH_DIR/${synth_folder}/${input_name}__${model}.csv"
        if [[ ! -f "$synth_file" ]]; then
            # Try alternate naming (lowercase dataset in filename)
            synth_file="$SYNTH_DIR/${synth_folder}/${synth_folder}__${model}.csv"
        fi

        # Output file path
        output_file="$OUTPUT_DIR/benchmark__${input_name}__${model}.json"

        # Check if synthetic file exists
        if [[ ! -f "$synth_file" ]]; then
            echo "  [SKIP] $model - synthetic file not found"
            skipped=$((skipped + 1))
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
echo "Skipped: $skipped"
echo "Failed:  $failed"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
