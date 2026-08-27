#!/bin/bash
#SBATCH --job-name=csi-launch-sweep
#SBATCH --partition=all
#SBATCH --account=princeton
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/launch-sweep-%j.out
#SBATCH --error=logs/launch-sweep-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO_ROOT/.venv/bin/python"
DRIVER="$SCRIPT_DIR/expectation_fit_study3.2_real_sequences_single_csi_leak12.py"
DATA_FILE="$SCRIPT_DIR/data_to_fit_study3.csv"
RESULT_DIR="$SCRIPT_DIR/logs/launch-sweep-${SLURM_JOB_ID}"
SUMMARY="$RESULT_DIR/summary.tsv"

mkdir -p "$RESULT_DIR"
cd "$SCRIPT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$DATA_FILE" ]]; then
    echo "Behavioral data not found: $DATA_FILE" >&2
    exit 1
fi

# Fail clearly when the sweep script has been copied without the matching
# fit-driver changes. argparse exits with status 2 for unknown options, which
# would otherwise leave every row in the summary filled with NaNs.
driver_help="$($PYTHON "$DRIVER" --help 2>&1)" || {
    echo "Unable to load fit driver: $DRIVER" >&2
    echo "$driver_help" >&2
    exit 1
}
for required_option in \
    --triton-block-size \
    --triton-num-warps \
    --triton-maxnreg \
    --skip-fit-output; do
    if [[ "$driver_help" != *"$required_option"* ]]; then
        echo "Fit driver does not support $required_option." >&2
        echo "Update the fit driver and batched Triton implementation to the same revision as this sweep script." >&2
        exit 1
    fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# label:block_size:num_warps:maxnreg. "none" preserves unconstrained registers.
# The repeated baseline detects clock/thermal drift across the sweep.
CONFIGS=(
    "baseline_a:128:4:none"
    "b32_w1:32:1:none"
    "b64_w2:64:2:none"
    "b64_w4:64:4:none"
    "b128_w2:128:2:none"
    "b128_w8:128:8:none"
    "b256_w4:256:4:none"
    "b256_w8:256:8:none"
    "b128_w4_r128:128:4:128"
    "b128_w4_r112:128:4:112"
    "b128_w4_r96:128:4:96"
    "b128_w4_r80:128:4:80"
    "b128_w4_r64:128:4:64"
    "baseline_b:128:4:none"
)

printf "label\tblock\twarps\tmaxnreg\tactual_regs\tstack_bytes\tlocal_bytes\tshared_bytes\twarm_batches\tmean_s\tmin_s\tmax_s\tparameter_sets_per_s\ttotal_s\tstatus\toptimal_ll\n" > "$SUMMARY"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Results: $RESULT_DIR"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

for config in "${CONFIGS[@]}"; do
    IFS=: read -r label block_size num_warps maxnreg <<< "$config"
    run_log="$RESULT_DIR/${label}.log"
    triton_cache="$RESULT_DIR/triton-cache-${label}"
    mkdir -p "$triton_cache"
    extra_args=()
    if [[ "$maxnreg" != "none" ]]; then
        extra_args+=(--triton-maxnreg "$maxnreg")
    fi

    echo
    echo "Running $label: block_size=$block_size num_warps=$num_warps maxnreg=$maxnreg"
    start_ns="$(date +%s%N)"
    set +e
    TRITON_CACHE_DIR="$triton_cache" "$PYTHON" -u "$DRIVER" \
        --backend triton \
        --subject_id 1 \
        --data_file "$DATA_FILE" \
        --num_estimates 10000 \
        --max_iterations 111 \
        --max_steps 1200 \
        --bins 100 \
        --parameter_batch_size 11 \
        --cpu_count "${SLURM_CPUS_PER_TASK:-8}" \
        --seed 1 \
        --triton-block-size "$block_size" \
        --triton-num-warps "$num_warps" \
        --skip-posterior-predictive \
        --skip-fit-output \
        "${extra_args[@]}" > "$run_log" 2>&1
    status=$?
    set -e
    end_ns="$(date +%s%N)"

    if (( status != 0 )); then
        echo "$label failed with status $status; last 20 log lines:" >&2
        tail -n 20 "$run_log" >&2
    fi

    total_s="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.3f", (end-start)/1000000000 }')"
    stats="$(awk '
        $1 == "Evaluated" && $2 == "11" && $5 == "Batch-Eval-Time:" {
            full_count++;
            # The first full-sized launch JIT-compiles this specialization.
            # Exclude it so configurations are compared at steady state.
            if (full_count == 1) next;
            value=$6; count++; sum+=value;
            if (count == 1 || value < min) min=value;
            if (count == 1 || value > max) max=value;
        }
        END {
            if (count) printf "%d %.6f %.6f %.6f %.3f", count, sum/count, min, max, 11/(sum/count);
            else printf "0 nan nan nan nan";
        }
    ' "$run_log")"
    read -r warm_batches mean_s min_s max_s sets_per_s <<< "$stats"
    optimal_ll="$(awk '/^Optimal Log-Likelihood:/ { value=$3 } END { print value == "" ? "nan" : value }' "$run_log")"

    actual_regs=nan
    stack_bytes=nan
    local_bytes=nan
    shared_bytes=nan
    cubin="$(find "$triton_cache" -type f -name 'pnl_batched_coevolving_graph_kernel.cubin' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -n "$cubin" ]] && command -v cuobjdump >/dev/null 2>&1; then
        resource_line="$(cuobjdump --dump-resource-usage "$cubin" 2>/dev/null | awk '
            /Function pnl_batched_coevolving_graph_kernel:/ { getline; print; exit }
        ')"
        for token in $resource_line; do
            case "$token" in
                REG:*) actual_regs="${token#REG:}" ;;
                STACK:*) stack_bytes="${token#STACK:}" ;;
                LOCAL:*) local_bytes="${token#LOCAL:}" ;;
                SHARED:*) shared_bytes="${token#SHARED:}" ;;
            esac
        done
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$label" "$block_size" "$num_warps" "$maxnreg" "$actual_regs" \
        "$stack_bytes" "$local_bytes" "$shared_bytes" "$warm_batches" \
        "$mean_s" "$min_s" "$max_s" "$sets_per_s" "$total_s" "$status" \
        "$optimal_ll" >> "$SUMMARY"

    echo "$label: mean=${mean_s}s, parameter sets/s=${sets_per_s}, registers=${actual_regs}, stack=${stack_bytes}, local=${local_bytes}, status=$status"
done

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "Summary: $SUMMARY"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$SUMMARY"
else
    cat "$SUMMARY"
fi
