#!/usr/bin/env bash
# Run synthony-benchmark for all datasets in trial1
# Initialize conda if available
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    conda activate table-synthesizers
fi

set -e

# Directories
INPUT_DIR="./dataset/input_data"
SYNTH_DIR="./dataset/synth_data/spark"
OUTPUT_DIR="./output/benchmark/spark"

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
    "CTGAN"
    "DPCART"
    "Identity"
    "NFlow:nflow"
    "PATECTGAN"
    "TAPDDPM"
    "SMOTE"
    "TabDDPM"
    "TabSyn:tabsyn"
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

        # Support mapping like "DisplayName:filename_suffix".
        if [[ "$model" == *":"* ]]; then
            model_display="${model%%:*}"
            model_suffix="${model##*:}"
        else
            model_display="$model"
            # use lowercase of the model name as the file suffix (matches existing files)
            model_suffix="$(echo "$model" | tr '[:upper:]' '[:lower:]')"
        fi

        # Synthetic file path - try two naming conventions
        synth_file="$SYNTH_DIR/${synth_folder}/${input_name}_synthetic_${model_suffix}_1000.csv"
        if [[ ! -f "$synth_file" ]]; then
            synth_file="$SYNTH_DIR/${synth_folder}/${synth_folder}_synthetic_${model_suffix}_1000.csv"
        fi

        # Output file path (use display name for clarity)
        safe_model_name="${model_display//[^a-zA-Z0-9_]/_}"
        output_file="$OUTPUT_DIR/benchmark__${input_name}__${safe_model_name}.json"

        # Check if synthetic file exists
        if [[ ! -f "$synth_file" ]]; then
            echo "  [SKIP] $model_display - synthetic file not found"
            skipped=$((skipped + 1))
            continue
        fi

        echo -n "  Benchmarking $model_display... "

        # Run benchmark using synthony-benchmark (use full path and verbose)
        if /opt/anaconda3/bin/synthony-benchmark -v -r "$original_file" -s "$synth_file" -o "$output_file"; then
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
