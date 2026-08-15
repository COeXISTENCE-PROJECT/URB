#!/usr/bin/env bash

set -euo pipefail

# These experiments run unattended, so plotting must not use the macOS GUI
# backend when SUMO or metrics finish in a background session.
export MPLBACKEND=Agg

if [[ $# -ne 2 ]]; then
    printf 'Usage: %s <seed> <janux-first|clustered-first>\n' "$0" >&2
    exit 2
fi

seed="$1"
order="$2"

case "$order" in
    janux-first)
        route_order=(janux clustered)
        ;;
    clustered-first)
        route_order=(clustered janux)
        ;;
    *)
        printf 'Unknown order: %s\n' "$order" >&2
        exit 2
        ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="$repo_root/venv/bin/python"
results_dir="$repo_root/studies/road-clustering-improves-iql/results"
network="ingolstadt_custom2"
task_config="selfish_40"
route_set="default-pre-integration"

export RESULTS_BASE_DIR="$results_dir"

if [[ ! -x "$python_bin" ]]; then
    printf 'Project Python not found: %s\n' "$python_bin" >&2
    exit 1
fi

run_metrics() {
    local experiment_id="$1"
    "$python_bin" "$repo_root/analysis/metrics.py" \
        --id "$experiment_id" \
        --results-folder "$results_dir"
}

resume_if_finished() {
    local experiment_id="$1"
    local final_episode="$2"
    local experiment_dir="$results_dir/$experiment_id"
    local benchmark_file="$experiment_dir/metrics/BenchmarkMetrics.csv"
    local vector_file="$experiment_dir/metrics/VectorMetrics.csv"
    local metric_plots="$experiment_dir/metrics/plots"

    if [[ -s "$benchmark_file" \
        && -s "$vector_file" \
        && -f "$metric_plots/action change count.png" \
        && -f "$metric_plots/avg time lost.png" \
        && -f "$metric_plots/time excess.png" ]]; then
        printf '[%s] Skipping completed experiment %s\n' \
            "$(date -u +%FT%TZ)" "$experiment_id"
        return 0
    fi

    if [[ -f "$experiment_dir/episodes/ep${final_episode}.csv" ]]; then
        printf '[%s] Recovering metrics for finished experiment %s\n' \
            "$(date -u +%FT%TZ)" "$experiment_id"
        run_metrics "$experiment_id"
        return 0
    fi

    if [[ -e "$experiment_dir" ]]; then
        printf 'Refusing to overwrite incomplete experiment: %s\n' \
            "$experiment_dir" >&2
        return 1
    fi

    return 2
}

run_aon() {
    local route_kind="$1"
    local experiment_id="rciql_ing2_s${seed}_aon_${route_kind}"
    local args=(
        "$python_bin" "$repo_root/scripts/baselines_clusters.py"
        --id "$experiment_id"
        --alg-conf config1
        --task-conf "$task_config"
        --net "$network"
        --env-seed "$seed"
        --model aon
    )

    if resume_if_finished "$experiment_id" 600; then
        return
    else
        local resume_status=$?
        if [[ "$resume_status" -ne 2 ]]; then
            return "$resume_status"
        fi
    fi

    if [[ "$route_kind" == "clustered" ]]; then
        args+=(--env-conf clusters --route-set "$route_set")
    else
        args+=(--env-conf config1)
    fi

    printf '\n[%s] Starting AON/%s\n' "$(date -u +%FT%TZ)" "$route_kind"
    PYTHONHASHSEED="$seed" "${args[@]}"
    run_metrics "$experiment_id"
}

run_iql() {
    local route_kind="$1"
    local experiment_id="rciql_ing2_s${seed}_iql_${route_kind}"
    local args=(
        "$python_bin" "$repo_root/scripts/iql_clusters.py"
        --id "$experiment_id"
        --alg-conf config1
        --task-conf "$task_config"
        --net "$network"
        --env-seed "$seed"
        --torch-seed "$seed"
    )

    if resume_if_finished "$experiment_id" 4400; then
        return
    else
        local resume_status=$?
        if [[ "$resume_status" -ne 2 ]]; then
            return "$resume_status"
        fi
    fi

    if [[ "$route_kind" == "clustered" ]]; then
        args+=(--env-conf clusters --route-set "$route_set")
    else
        args+=(--env-conf config1)
    fi

    printf '\n[%s] Starting IQL/%s\n' "$(date -u +%FT%TZ)" "$route_kind"
    PYTHONHASHSEED="$seed" "${args[@]}"
    run_metrics "$experiment_id"
}

printf '[%s] Starting replication seed=%s order=%s\n' "$(date -u +%FT%TZ)" "$seed" "$order"

for route_kind in "${route_order[@]}"; do
    run_aon "$route_kind"
done

for route_kind in "${route_order[@]}"; do
    run_iql "$route_kind"
done

printf '\n[%s] Replication seed=%s complete\n' "$(date -u +%FT%TZ)" "$seed"
