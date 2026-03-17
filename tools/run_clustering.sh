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
### Results are published as a GitHub Release (requires gh CLI).
###
### --image_name  Docker image name with tag built from the provided Dockerfile
### --output_dir  Top-level directory for intermediate calculations; species
###               subdirectories are created automatically. Previous .npy files
###               are preserved across runs unless --clean is passed.
### --cpus        Number of threads for Numba parallel distance computation
### --clean       Force full recalculation (removes cached distance matrices)
###
### Script will crash if machine has less than 600 Gb of RAM
### Example:
### ./tools/run_clustering.sh --output_dir /mnt/raid/michall/pHierCC \
###     --image_name "phiercc_custom:2.0" --cpus 250

output_dir=""
image_name=""
cpus=1
clean=false

function show_help() {
    echo "Usage: $0 --output_dir <path> --image_name <string> --cpus <int> [--clean]"
    echo ""
    echo "Options:"
    echo "  --output_dir   Path to top-level directory for calculations"
    echo "  --image_name   Docker image name:tag built from the Dockerfile"
    echo "  --cpus         Number of CPUs/threads (default: 1)"
    echo "  --clean        Force full recalculation (remove cached .npy files)"
    echo "  -h, --help     Show this help message"
}

OPTIONS=$(getopt -o h --long output_dir:,image_name:,cpus:,clean,help -- "$@")
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
# Prepare output directories
# ---------------------------------------------------------------------------
output=$(realpath "${output_dir}")

for species in Salmonella Escherichia Campylobacter; do
    if [ ! -d "${output}/${species}" ]; then
        mkdir -p "${output}/${species}"
    else
        # Remove old profile downloads (new ones will be fetched below)
        rm -f "${output}/${species}"/profiles.list*

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
# Download profiles
# ---------------------------------------------------------------------------
wget -O "${output}/Salmonella/profiles.list.gz"  "https://enterobase.warwick.ac.uk//schemes/Salmonella.cgMLSTv2/profiles.list.gz"
wget -O "${output}/Escherichia/profiles.list.gz" "https://enterobase.warwick.ac.uk//schemes/Escherichia.cgMLSTv1/profiles.list.gz"
python3 tools/download_profile_Campylo.py -o "${output}/Campylobacter/profiles.list"

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

for species in Campylobacter Escherichia Salmonella; do
    profile_file="profiles.list.gz"
    if [ "$species" = "Campylobacter" ]; then
        profile_file="profiles.list"
    fi

    echo "Running clustering for ${species} on ${cpus} CPUs"
    set +e
    docker run --rm \
           --volume "${output}/${species}/:/dane:rw" \
           --user "$(id -u):$(id -g)" \
           --ulimit nofile=262144:262144 \
           ${image_name} --profile "/dane/${profile_file}" -n ${cpus} \
           --clustering_method single --clustering_method complete \
           ${clean_flag}
    rc=$?
    set -e

    if [ "$rc" -eq 42 ]; then
        echo "No new STs for ${species}, skipping."
        continue
    elif [ "$rc" -ne 0 ]; then
        echo "pHierCC failed for ${species} (exit ${rc})"
        exit "$rc"
    fi

    any_updated=true
    echo "Finished calculations for ${species}"
done

# ---------------------------------------------------------------------------
# Publish results as a GitHub Release (only if at least one species updated)
# ---------------------------------------------------------------------------
if [ "$any_updated" = true ]; then
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
