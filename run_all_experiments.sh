#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_all_experiments.sh
#
# Run from project root:
#   bash run_all_experiments.sh
# ==============================================================================

CONFIGS=(
  "configs/periodic_heat.toml"
  "configs/dirichlet_heat.toml"
  "configs/periodic_poisson.toml"
  "configs/variable_poisson.toml"
)

MODEL_CONDITIONS=(
  "fno false"
  "fno true"
  "deeponet false"
)

SUMMARY_PATH="results/tables/summary.csv"
LOG_DIR="results/logs"

mkdir -p "$LOG_DIR"
mkdir -p results/tables
mkdir -p data/raw
mkdir -p checkpoints/fno
mkdir -p checkpoints/deeponet

echo "================================================="
echo " FULL FNO / DeepONet PDE EXPERIMENT"
echo " Running from project root: $(pwd)"
echo "================================================="

# Start fresh summary.
if [ -f "$SUMMARY_PATH" ]; then
  echo "Removing existing summary: $SUMMARY_PATH"
  rm "$SUMMARY_PATH"
fi

echo ""
echo "================================================="
echo " 1. GENERATING DATASETS"
echo "================================================="

for config in "${CONFIGS[@]}"; do
  dataset_name="$(basename "$config" .toml)"
  log_file="$LOG_DIR/generate_${dataset_name}.log"

  echo ""
  echo "Generating dataset: $config"
  echo "Log: $log_file"

  julia --project=. scripts/generate_data.jl "$config" 2>&1 | tee "$log_file"
done

echo ""
echo "================================================="
echo " 2. TRAINING AND EVALUATING MODELS"
echo "================================================="

for config in "${CONFIGS[@]}"; do
  dataset_name="$(basename "$config" .toml)"

  for condition in "${MODEL_CONDITIONS[@]}"; do
    model_name="$(echo "$condition" | awk '{print $1}')"
    use_coord="$(echo "$condition" | awk '{print $2}')"

    train_log="$LOG_DIR/train_${dataset_name}_${model_name}_${use_coord}.log"
    eval_log="$LOG_DIR/eval_${dataset_name}_${model_name}_${use_coord}.log"

    echo ""
    echo "-------------------------------------------------"
    echo "Dataset:   $dataset_name"
    echo "Model:     $model_name"
    echo "Use coord: $use_coord"
    echo "-------------------------------------------------"

    echo "Training..."
    julia --project=. scripts/train_one.jl "$config" "$model_name" "$use_coord" 2>&1 | tee "$train_log"

    echo "Evaluating..."
    julia --project=. scripts/evaluate_one.jl "$config" "$model_name" "$use_coord" 2>&1 | tee "$eval_log"
  done
done

echo ""
echo "================================================="
echo " EXPERIMENT COMPLETE"
echo "================================================="
echo "Summary CSV: $SUMMARY_PATH"
echo "Logs:        $LOG_DIR"

if [ -f "$SUMMARY_PATH" ]; then
  echo ""
  echo "Summary preview:"
  cat "$SUMMARY_PATH"
fi