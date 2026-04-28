#!/bin/bash
set -euo pipefail
### Script downloads cgMLST profiles for Campylobacter, Salmonella and Ecoli
### and performs clustering using pHierCC methodology.
###
### By default, calculations are incremental: distance matrices (dist0.npy,
### dist1.npy) and ordering (ordering.npy) from a previous run are reused,
### and only pairs involving new STs are computed. Pass --clean to force a
### full recalculation from scratch.
###
### When --gpu-ids is provided, distance matrices are computed on CUDA GPUs
### (always a full recalculation). The Docker container is started with
### --gpus to expose the requested devices.
###
### Results are published as a GitHub Release (requires gh CLI).
###
### --image_name  Docker image name with tag built from the provided Dockerfile
### --output_dir  Top-level directory for intermediate calculations; species
###               subdirectories are created automatically. Previous .npy files
###               are preserved across runs unless --clean is passed.
### --cpus        Number of threads for Numba parallel distance computation
### --clean       Force full recalculation (removes cached distance matrices)
### --gpu-ids     Space-separated CUDA device IDs (e.g. "0 1 2 3") or "all"
###               to auto-detect. When set, GPU acceleration is used and
###               --cpus controls only clustering.
### --block-size  Tile edge size for GPU computation (default: 100000)
###
### Script will crash if machine has less than 600 Gb of RAM
### Example (CPU):
### ./tools/run_clustering.sh --output_dir /mnt/raid/michall/pHierCC \
###     --image_name "plepiseq-cluster:3.0" --cpus 250
### Example (GPU):
### ./tools/run_clustering.sh --output_dir /mnt/raid/michall/pHierCC \
###     --image_name "plepiseq-cluster:3.0" --cpus 1 --gpu-ids "0 1 2 3"

output_dir=""
image_name=""
cpus=1
clean=false
gpu_ids=""
block_size=""
salmonella_local=""
escherichia_local=""
campylobacter_local=""

function show_help() {
    echo "Usage: $0 --output_dir <path> --image_name <string> --cpus <int> [--clean] [--gpu-ids \"0 1 ...\"] [--block-size N] [--salmonella-local PATH] [--escherichia-local PATH] [--campylobacter-local PATH]"
    echo ""
    echo "Options:"
    echo "  --output_dir           Path to top-level directory for calculations"
    echo "  --image_name           Docker image name:tag built from the Dockerfile"
    echo "  --cpus                 Number of CPUs/threads (default: 1)"
    echo "  --clean                Force full recalculation (remove cached .npy files)"
    echo "  --gpu-ids              GPU device IDs (e.g. \"0 1 2 3\") or \"all\" to auto-detect"
    echo "  --block-size           Tile edge size for GPU computation (default: 100000)"
    echo "  --salmonella-local     Path to plain-text Salmonella local_* profile (optional)"
    echo "  --escherichia-local    Path to plain-text Escherichia local_* profile (optional)"
    echo "  --campylobacter-local  Path to plain-text Campylobacter local_* profile (optional)"
    echo "  -h, --help             Show this help message"
}

OPTIONS=$(getopt -o h --long output_dir:,image_name:,cpus:,clean,gpu-ids:,block-size:,salmonella-local:,escherichia-local:,campylobacter-local:,help -- "$@")
eval set -- "$OPTIONS"

if [[ $# -eq 1 ]]; then
    echo "No parameters provided"
    show_help
    exit 1
fi

while true; do
    case "$1" in
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --cpus)
            cpus="$2"
            shift 2
            ;;
        --image_name)
            image_name="$2"
            shift 2
            ;;
        --clean)
            clean=true
            shift
            ;;
        --gpu-ids)
            gpu_ids="$2"
            shift 2
            ;;
        --block-size)
            block_size="$2"
            shift 2
            ;;
        --salmonella-local)
            salmonella_local="$2"
            shift 2
            ;;
        --escherichia-local)
            escherichia_local="$2"
            shift 2
            ;;
        --campylobacter-local)
            campylobacter_local="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Sanity check: tools/ must be reachable (for download_profile_Campylo.py)
if [ ! -d tools ]; then
    echo "Error: tools/ not found. Run this script from the repository root."
    show_help
    exit 1
fi

## Validate required arguments
if [[ -z "$output_dir" ]]; then
    echo "Error: --output_dir is required."
    show_help
    exit 1
fi

if [[ -z "$image_name" ]]; then
    echo "Error: --image_name is required."
    show_help
    exit 1
fi

## Verify Docker image exists
tmp_name=$(echo "${image_name}" | cut -d ":" -f1)
tmp_tag=$(echo "${image_name}" | cut -d ":" -f2)

if [ "$(docker images | grep "${tmp_name}" | grep "${tmp_tag}" | wc -l)" -ne 1 ]; then
    echo "Provided docker image ${tmp_name}:${tmp_tag} does not exist. Provide valid image name"
    exit 1
fi

## Verify gh CLI is available (needed for publishing releases)
if ! command -v gh &>/dev/null; then
    echo "Error: gh CLI not found. Install it from https://cli.github.com/"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve --gpu-ids "all" to actual device IDs via nvidia-smi
# ---------------------------------------------------------------------------
gpu_all=false
if [[ "$gpu_ids" == "all" ]]; then
    gpu_all=true
    if ! command -v nvidia-smi &>/dev/null; then
        echo "Error: --gpu-ids all requires nvidia-smi to detect devices."
        exit 1
    fi
    n_gpus=$(nvidia-smi --list-gpus | wc -l)
    if [[ "$n_gpus" -lt 1 ]]; then
        echo "Error: nvidia-smi found 0 GPUs."
        exit 1
    fi
    gpu_ids=$(seq -s ' ' 0 $((n_gpus - 1)))
    echo "Detected ${n_gpus} GPUs: ${gpu_ids}"
fi

# ---------------------------------------------------------------------------
# Build pHierCC GPU args (passed inside docker run) -- bash array
# ---------------------------------------------------------------------------
phiercc_gpu_args=()
if [[ -n "$gpu_ids" ]]; then
    for gid in $gpu_ids; do
        phiercc_gpu_args+=(--gpu-ids "$gid")
    done
    if [[ -n "$block_size" ]]; then
        phiercc_gpu_args+=(--block-size "$block_size")
    fi
fi

# ---------------------------------------------------------------------------
# Build docker GPU args (passed to docker run itself) -- bash array
# Use --gpus all when user requested "all" to avoid the NVIDIA runtime
# "cannot set both Count and DeviceIDs" error that occurs when all
# device IDs are enumerated explicitly.
# ---------------------------------------------------------------------------
docker_gpu_args=()
if [[ -n "$gpu_ids" ]]; then
    if [[ "$gpu_all" == true ]]; then
        docker_gpu_args+=(--gpus all)
    else
        device_list=$(echo "$gpu_ids" | tr ' ' ',')
        docker_gpu_args+=(--gpus "device=${device_list}")
    fi
fi

# ---------------------------------------------------------------------------
# Map species -> user-supplied local profile path (empty = none)
# ---------------------------------------------------------------------------
declare -A local_profile=(
    [Salmonella]="$salmonella_local"
    [Escherichia]="$escherichia_local"
    [Campylobacter]="$campylobacter_local"
)

# Validate any provided local files exist and are readable
for species in Salmonella Escherichia Campylobacter; do
    local_src="${local_profile[$species]}"
    if [[ -n "$local_src" && ! -r "$local_src" ]]; then
        echo "Error: --${species,,}-local path does not exist or is not readable: $local_src"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Prepare output directories
# ---------------------------------------------------------------------------
output=$(realpath "${output_dir}")

for species in Salmonella Escherichia Campylobacter; do
    if [ ! -d "${output}/${species}" ]; then
        mkdir -p "${output}/${species}"
    else
        # Remove old profile downloads and staging files (fresh copies
        # will be produced below). profiles.list[.gz] = post-merge file
        # consumed by pHierCC; profiles_external.list[.gz] = raw download;
        # profiles_local.list = copy of the user-supplied local profile.
        rm -f "${output}/${species}"/profiles.list*
        rm -f "${output}/${species}"/profiles_external.list*
        rm -f "${output}/${species}"/profiles_local.list

        if [ "$clean" = true ]; then
            echo "--clean: removing cached distance matrices for ${species}"
            rm -f "${output}/${species}"/dist0.npy
            rm -f "${output}/${species}"/dist1.npy
            rm -f "${output}/${species}"/ordering.npy
        fi
    fi
done

if [ ! -w "$output" ]; then
    echo "Current user does not have write permissions to the directory $output"
    exit 1
fi

# ---------------------------------------------------------------------------
# Download external profiles (retry once after 300 seconds on failure)
# ---------------------------------------------------------------------------
retry_download() {
    local label="$1"; shift
    local wait_s=300
    if ! "$@"; then
        echo "WARNING: ${label} download failed; retrying in ${wait_s}s"
        sleep "${wait_s}"
        if ! "$@"; then
            echo "ERROR: ${label} download failed after retry"
            return 1
        fi
    fi
}

retry_download "Salmonella" \
    wget -q --tries=1 --timeout=60 \
    -O "${output}/Salmonella/profiles_external.list.gz" \
    "https://enterobase.warwick.ac.uk//schemes/Salmonella.cgMLSTv2/profiles.list.gz" \
    || exit 1
retry_download "Escherichia" \
    wget -q --tries=1 --timeout=60 \
    -O "${output}/Escherichia/profiles_external.list.gz" \
    "https://enterobase.warwick.ac.uk//schemes/Escherichia.cgMLSTv1/profiles.list.gz" \
    || exit 1
retry_download "Campylobacter" \
    python3 tools/download_profile_Campylo.py \
    -o "${output}/Campylobacter/profiles_external.list" \
    || exit 1

# ---------------------------------------------------------------------------
# Merge external + (optional) local profile into the segmented
# profiles.list[.gz] that pHierCC will consume.
#
# Contract: numeric STs first in ascending order, then local_* STs in
# ascending suffix order. This avoids issue #8 / bug 2 (mixed/interleaved
# inputs produce broken .HierCC.index segmentation).
# ---------------------------------------------------------------------------
read_profile() {
    local path="$1"
    if [[ "$path" == *.gz ]]; then
        zcat "$path"
    else
        cat "$path"
    fi
}

# Read just the first line of a (possibly gzipped) file. Drains the rest of
# the stream into /dev/null so the upstream decompressor never receives
# SIGPIPE -- which together with `set -euo pipefail` would otherwise abort
# the whole script at the first header read.
read_header_line() {
    local path="$1"
    if [[ "$path" == *.gz ]]; then
        zcat "$path" | { IFS= read -r line; cat > /dev/null; printf '%s\n' "$line"; }
    else
        IFS= read -r line < "$path"
        printf '%s\n' "$line"
    fi
}

prepare_profile() {
    local species="$1"
    local species_dir="${output}/${species}"
    local local_src="${local_profile[$species]}"

    local external_path="${species_dir}/profiles_external.list.gz"
    local final_path="${species_dir}/profiles.list.gz"
    if [ "$species" = "Campylobacter" ]; then
        external_path="${species_dir}/profiles_external.list"
        final_path="${species_dir}/profiles.list"
    fi

    local ext_header
    ext_header=$(read_header_line "$external_path")
    if [ -z "$ext_header" ]; then
        echo "ERROR: external profile for ${species} is empty (${external_path})"
        exit 1
    fi

    local tmp_body
    tmp_body=$(mktemp)

    local local_is_header_only=false
    if [ -n "$local_src" ]; then
        # Reject empty files early: 0 bytes is almost always a broken upstream
        # export. A header-only file, in contrast, is a legitimate "no new
        # local STs this week" state and only warrants a warning.
        if [ ! -s "$local_src" ]; then
            echo "ERROR: ${species} local profile is empty (${local_src})"
            rm -f "$tmp_body"
            exit 1
        fi

        cp "$local_src" "${species_dir}/profiles_local.list"
        local local_path="${species_dir}/profiles_local.list"

        local loc_header
        loc_header=$(head -n 1 "$local_path")
        if [ "$loc_header" != "$ext_header" ]; then
            echo "ERROR: header mismatch between external and local profile for ${species}"
            echo "  external: ${ext_header:0:120}..."
            echo "  local:    ${loc_header:0:120}..."
            rm -f "$tmp_body"
            exit 1
        fi

        local loc_data_rows
        loc_data_rows=$(($(wc -l < "$local_path") - 1))
        if [ "$loc_data_rows" -le 0 ]; then
            echo "WARNING: ${species} local profile has only a header, no local_* rows (${local_src}); proceeding with external profile only"
            local_is_header_only=true
        else
            local bad_row
            bad_row=$(tail -n +2 "$local_path" \
                | awk -F'\t' '$1 !~ /^local_[0-9]+$/ {print NR": "$1; exit}')
            if [ -n "$bad_row" ]; then
                echo "ERROR: ${species} local profile contains non-local_* ST at data row ${bad_row}"
                rm -f "$tmp_body"
                exit 1
            fi
        fi
    fi

    # Header + external rows (numeric-ascending) + local rows (local_N ascending)
    #
    # We do NOT trust either stream to contain exactly one header. Instead of
    # `tail -n +2`, we filter by the first column: any row whose first cell
    # equals the canonical header label (e.g. "cgST") is dropped. This defends
    # against duplicate header rows in the download (observed on Campylobacter
    # April 2026 -- origin never reproduced: likely a transient PubMLST
    # response or a mid-read overwrite) and against future upstream mistakes.
    # The canonical header is written exactly once, from `$ext_header`.
    local hdr1="${ext_header%%$'\t'*}"

    printf '%s\n' "$ext_header" > "$tmp_body"
    read_profile "$external_path" \
        | awk -v h="$hdr1" -F'\t' 'NR>1 && $1 != h' \
        | sort -t$'\t' -k1,1n \
        >> "$tmp_body"

    local n_ext n_loc=0
    n_ext=$(read_profile "$external_path" \
        | awk -v h="$hdr1" -F'\t' 'NR>1 && $1 != h' \
        | wc -l)

    if [ -n "$local_src" ] && [ "$local_is_header_only" = false ]; then
        local local_path="${species_dir}/profiles_local.list"
        awk -v h="$hdr1" -F'\t' 'NR>1 && $1 != h' "$local_path" \
            | sort -t_ -k2,2n \
            >> "$tmp_body"
        n_loc=$(awk -v h="$hdr1" -F'\t' 'NR>1 && $1 != h' "$local_path" | wc -l)
    fi

    local n_total=$((n_ext + n_loc))

    # Final safety net: the merged file must contain exactly one header row
    # (first column equal to the canonical header label). Anything else is a
    # bug upstream in this function; abort before clustering wastes hours on
    # a corrupt profile.
    local n_hdr
    n_hdr=$(awk -v h="$hdr1" -F'\t' '$1 == h {c++} END {print c+0}' "$tmp_body")
    if [ "$n_hdr" -ne 1 ]; then
        echo "ERROR: merged ${species} profile contains ${n_hdr} header rows (expected 1)"
        rm -f "$tmp_body"
        exit 1
    fi

    if [[ "$final_path" == *.gz ]]; then
        gzip -n -c "$tmp_body" > "$final_path"
        rm -f "$tmp_body"
    else
        mv "$tmp_body" "$final_path"
    fi

    echo "Prepared ${species} profile: ${n_ext} external + ${n_loc} local = ${n_total} STs"
}

for species in Salmonella Escherichia Campylobacter; do
    prepare_profile "$species"
done

TIMESTAMP=$(date +%Y-%m-%d)

# ---------------------------------------------------------------------------
# Build the --clean flag string for docker commands
# ---------------------------------------------------------------------------
clean_flag=""
if [ "$clean" = true ]; then
    clean_flag="--clean"
fi

# ---------------------------------------------------------------------------
# Clustering
# pHierCC exits with code 42 when the profile is unchanged (no new STs).
# We track whether at least one species was updated to decide if a release
# should be created.
# ---------------------------------------------------------------------------
any_updated=false
any_failed=false

for species in Campylobacter Escherichia Salmonella; do
    profile_file="profiles.list.gz"
    if [ "$species" = "Campylobacter" ]; then
        profile_file="profiles.list"
    fi

    echo "Running clustering for ${species} on ${cpus} CPUs"

    cmd=(docker run --rm)
    if [[ -n "$gpu_ids" ]]; then
        cmd+=("${docker_gpu_args[@]}")
    else
        cmd+=(--ulimit nofile=262144:262144)
    fi
    cmd+=(--volume "${output}/${species}/:/dane:rw")
    cmd+=(--user "$(id -u):$(id -g)")
    cmd+=("${image_name}")
    cmd+=(--profile "/dane/${profile_file}" -n "${cpus}")
    cmd+=(--clustering_method single --clustering_method complete)
    if [[ -n "$clean_flag" ]]; then
        cmd+=("$clean_flag")
    fi
    if [[ ${#phiercc_gpu_args[@]} -gt 0 ]]; then
        cmd+=("${phiercc_gpu_args[@]}")
    fi

    set +e
    "${cmd[@]}"
    rc=$?
    set -e

    if [ "$rc" -eq 42 ]; then
        echo "No new STs for ${species}, skipping."
        continue
    elif [ "$rc" -ne 0 ]; then
        echo "pHierCC failed for ${species} (exit ${rc})"
        exit "$rc"
    fi

    # -------------------------------------------------------------------
    # Validate output files before marking this species as successful.
    # A truncated .gz or mismatched line count means the run produced
    # corrupt output. In that case we remove ordering.npy so the next
    # run won't skip, and we exclude this species from the release.
    # -------------------------------------------------------------------
    species_dir="${output}/${species}"
    profile_path="${species_dir}/${profile_file}"

    # Count expected STs from the input profile (lines minus header)
    if [[ "$profile_file" == *.gz ]]; then
        expected_sts=$(zcat "$profile_path" | wc -l)
    else
        expected_sts=$(wc -l < "$profile_path")
    fi
    expected_sts=$((expected_sts - 1))  # subtract header

    validation_ok=true
    for gz in "${species_dir}"/*_linkage.HierCC.gz; do
        basename_gz=$(basename "$gz")

        # 1. File must exist and be non-empty
        if [ ! -s "$gz" ]; then
            echo "VALIDATION FAILED: ${species}/${basename_gz} is missing or empty"
            validation_ok=false
            continue
        fi

        # 2. gzip integrity check
        if ! gzip -t "$gz" 2>/dev/null; then
            echo "VALIDATION FAILED: ${species}/${basename_gz} is a truncated/corrupt gzip"
            validation_ok=false
            continue
        fi

        # 3. Line count must match expected STs (data lines = header + STs)
        actual_lines=$(zcat "$gz" | wc -l)
        actual_sts=$((actual_lines - 1))  # subtract #ST_id header
        if [ "$actual_sts" -ne "$expected_sts" ]; then
            echo "VALIDATION FAILED: ${species}/${basename_gz} has ${actual_sts} STs, expected ${expected_sts}"
            validation_ok=false
        fi

        # 4. Corresponding .index file must exist
        idx="${gz%.gz}.index"
        if [ ! -s "$idx" ]; then
            echo "VALIDATION FAILED: ${species}/$(basename "$idx") is missing or empty"
            validation_ok=false
        fi
    done

    if [ "$validation_ok" = false ]; then
        echo "WARNING: Output validation failed for ${species}. Removing ordering.npy to force recalculation next run."
        rm -f "${species_dir}/ordering.npy"
        any_failed=true
        continue
    fi

    any_updated=true
    echo "Finished calculations for ${species} (${expected_sts} STs validated)"
done

# ---------------------------------------------------------------------------
# Publish results as a GitHub Release (only if at least one species updated
# AND none failed validation)
# ---------------------------------------------------------------------------
if [ "$any_failed" = true ]; then
    echo "WARNING: One or more species failed output validation. Skipping release."
    exit 1
elif [ "$any_updated" = true ]; then
    echo "Publishing results as GitHub Release v${TIMESTAMP}"

    release_dir=$(mktemp -d)

    for species in Salmonella Escherichia Campylobacter; do
        for f in "${output}/${species}"/*HierCC*; do
            cp "$f" "${release_dir}/${species}_$(basename "$f")"
        done
    done

    gh release create "v${TIMESTAMP}" \
        --title "Weekly clustering ${TIMESTAMP}" \
        --notes "Profiles downloaded on ${TIMESTAMP}." \
        "${release_dir}"/*

    rm -rf "${release_dir}"
    echo "Release v${TIMESTAMP} published successfully."
else
    echo "No species had new STs. Skipping release."
fi
