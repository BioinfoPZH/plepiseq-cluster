# plepiseq-cluster -- Hierarchical clustering of cgMLST profiles

This project is part of the PleEpiSeq project, co-funded by the European Union.

A modified version of [pHierCC](https://github.com/zheminzhou/pHierCC) (Zhou et al.) optimized for large-scale cgMLST hierarchical clustering (600,000+ sequence types, 3,002 loci) with significantly reduced RAM usage and faster distance computation.

---

## Table of contents

1. [Features](#features)
2. [Quick start](#quick-start)
3. [Requirements](#requirements)
4. [Running pHierCC](#running-phiercc)
5. [GPU acceleration](#gpu-acceleration)
6. [Clustering results](#clustering-results-github-releases)
7. [Repository structure](#repository-structure)
8. [Related projects](#related-projects)
9. [Citation](#citation)
10. [License](#license)

---

## Features

- **int16 distance matrices** -- Modified SciPy's `hierarchy.py` and `_hierarchy.pyx` to perform hierarchical clustering directly on `np.int16` distances, reducing RAM requirments 
- **Numba parallel threading** -- Replaced original parrarelization approach with Numba. This lowers memory overhead and imptove load balancing on triangular workloads.
- **Incremental distance computation** -- Add ability to reuse previously computed distance matrices to speed up calculation that include novel STs reducing calculation time.
- **Mixed ST identifiers** -- Support for both numeric and text-based sequence type identifiers.
- **Multi-GPU CUDA acceleration** -- Optional GPU path tiles the distance computation across multiple CUDA devices, reducing hours-long calculations to minutes.
- **Dockerized build** -- GPU (`Dockerfile`) and CPU-only (`Dockerfile.cpu`) images available.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/BioinfoPZH/plepiseq-cluster.git
cd plepiseq-cluster
```

### 2. Build the Docker image

```bash
docker build -t plepiseq-cluster:$(cat VERSION) -t plepiseq-cluster:latest .
```

### 3. Run on test data

```bash
docker run --rm \
    --volume $(pwd)/test_data:/dane:rw \
    --ulimit nofile=262144:262144 \
    plepiseq-cluster:latest \
    --profile /dane/profiles_19ST.list -n 4 --clustering_method single
```

Output files (`dist0.npy`, `dist1.npy`, `ordering.npy`, `profile_single_linkage.HierCC.gz`) will appear in `test_data/`.

---

## Requirements

| Category | Minimum | Recommended |
|----------|---------|-------------|
| **OS** | x86-64 Linux | Ubuntu 22.04 LTS or Debian 12 |
| **RAM** | 16 GB (small datasets) | >= 600 GB (Salmonella, 600k STs) |
| **CPUs** | 1 | 200+ for production workloads |

**Note:** The `--ulimit nofile=262144:262144` flag is required when running via Docker to prevent `OSError: [Errno 24] Too many open files` on large datasets.


---

## Running pHierCC

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `-p`, `--profile` | Path to tab-separated allelic profile file (can be gzipped) | *required* |
| `-n`, `--n_proc` | Number of threads for Numba parallel distance computation | 4 |
| `--clustering_method` | Linkage criterion (`single` / `complete`). Can be specified multiple times to run both in one invocation. | `single` |
| `-m`, `--allowed_missing` | Allowed proportion of missing genes in pairwise comparisons | 0.05 |
| `--clean` | Force full recalculation, removing cached distance matrices | `false` |
| `--gpu-ids` | CUDA GPU device IDs (can be repeated). Enables GPU distance computation; disables incremental mode. | *(disabled)* |
| `--block-size` | Tile edge size for GPU computation | 100000 |
| `--threads-per-block` | CUDA threads per block (two integers) | `16 16` |

### Full mode (first run)

When the working directory contains only the profile file, pHierCC computes all pairwise distances from scratch:

```bash
docker run --rm \
    --volume /path/to/workdir:/dane:rw \
    --user $(id -u):$(id -g) \
    --ulimit nofile=262144:262144 \
    plepiseq-cluster:latest \
    --profile /dane/profiles.list.gz -n 200 --clustering_method single
```

This produces:
- `dist0.npy` -- condensed distance matrix (squareform)
- `dist1.npy` -- full lower-triangular distance matrix
- `ordering.npy` -- ST ordering used during computation
- `profile_single_linkage.HierCC.gz` -- clustering results
- `profile_single_linkage.HierCC.index` -- index for fast lookups

### Incremental mode (subsequent runs)

When `dist0.npy`, `dist1.npy`, and `ordering.npy` exist in the working directory from a previous run, pHierCC automatically detects incremental mode. Only distances involving newly added STs are computed:

```bash
# Copy previous artefacts alongside the updated profile file
cp /previous_run/dist0.npy /previous_run/dist1.npy /previous_run/ordering.npy /path/to/workdir/

# Run with the new (larger) profile -- incremental mode activates automatically
docker run --rm \
    --volume /path/to/workdir:/dane:rw \
    --user $(id -u):$(id -g) \
    --ulimit nofile=262144:262144 \
    plepiseq-cluster:latest \
    --profile /dane/profiles_new.list.gz -n 200 --clustering_method single
```

Safeguards: if any old STs are missing from the new profile, or the cached distance matrix size does not match the expected number of STs, pHierCC falls back to full mode with a warning.

### Multiple clustering methods in one run

Pass `--clustering_method` more than once to perform several linkage methods without reloading the distance matrices:

```bash
docker run --rm \
    --volume /path/to/workdir:/dane:rw \
    --user $(id -u):$(id -g) \
    --ulimit nofile=262144:262144 \
    plepiseq-cluster:latest \
    --profile /dane/profiles.list.gz -n 200 \
    --clustering_method single --clustering_method complete
```

This produces both `profile_single_linkage.HierCC.gz` and `profile_complete_linkage.HierCC.gz` in a single invocation, avoiding the hours-long reload of distance matrices that a separate complete-linkage run would require.

### Forcing full recalculation

Use `--clean` to remove cached artefacts and recompute everything from scratch:

```bash
docker run --rm \
    --volume /path/to/workdir:/dane:rw \
    --user $(id -u):$(id -g) \
    --ulimit nofile=262144:262144 \
    plepiseq-cluster:latest \
    --profile /dane/profiles.list.gz -n 200 --clustering_method single --clean
```

---

## GPU acceleration

When CUDA GPUs are available, distance matrix computation can be offloaded to one or more devices using the `--gpu-ids` flag. GPU mode always performs a full recalculation (incremental mode is disabled) but is dramatically faster -- e.g. ~12 minutes vs. 2-3 hours for 395k STs on 7 GPUs.

### Docker prerequisites

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.
- Build the GPU-enabled image using the default `Dockerfile` (which includes `numba-cuda`):

```bash
docker build -t plepiseq-cluster:latest .
```

For machines **without** GPUs, build the CPU-only image:

```bash
docker build -f Dockerfile.cpu -t plepiseq-cluster:cpu .
```

### Running with GPUs

```bash
docker run --rm \
    --gpus '"device=0,1,2,3"' \
    --volume /path/to/workdir:/dane:rw \
    --user $(id -u):$(id -g) \
    plepiseq-cluster:latest \
    --profile /dane/profiles.list.gz -n 200 \
    --clustering_method single --clustering_method complete \
    --gpu-ids 0 --gpu-ids 1 --gpu-ids 2 --gpu-ids 3
```

The `--gpus` flag exposes host GPUs to the container; the `--gpu-ids` flags tell pHierCC which devices to use for distance tiling.

### Performance notes

- **Clustering** (linkage, attach genomes) still runs on CPUs -- only the distance matrices are GPU-accelerated.
- Tile size (`--block-size`) controls GPU memory usage per tile. The default of 100,000 works well for most setups; reduce it if you encounter out-of-memory errors.
- The `--threads-per-block` flag (default `16 16`) rarely needs tuning.

---

## Clustering results (GitHub Releases)

Pre-computed weekly clustering results for Salmonella, Escherichia, and Campylobacter are published as GitHub Release assets. To download the latest results:

```bash
gh release download --repo BioinfoPZH/plepiseq-cluster --pattern '*.gz' --dir ./clustering_data/
```

Or download a specific weekly snapshot:

```bash
gh release download v2026.03.04 --repo BioinfoPZH/plepiseq-cluster --pattern '*Salmonella*'
```

Releases follow the naming convention `vYYYY.MM.DD`, corresponding to the date when cgMLST profiles were downloaded from public databases.

---

## Repository structure

```
├── Dockerfile                      # GPU Docker image (SciPy + Numba + CUDA)
├── Dockerfile.cpu                  # CPU-only Docker image (no CUDA)
├── src/
│   ├── pHierCC.py                  # Main clustering script (CLI entrypoint)
│   └── getDistance.py              # Distance kernels (CPU Numba + CUDA GPU)
├── scipy_patches/                  # Modified SciPy cluster module (int16 support)
│   ├── hierarchy.py
│   └── _hierarchy.pyx
├── tools/
│   ├── run_clustering.sh           # Weekly automation wrapper (3 species)
│   ├── download_profile_Campylo.py # Campylobacter profile downloader
│   ├── compare_hiercc.py           # Compare two HierCC output files
│   └── test_incremental.py         # Incremental mode verification tests
├── test_data/                      # Small test profiles (9 and 19 STs)
├── VERSION
├── CHANGELOG.md
└── LICENSE                         # GPL-3.0
```

---

## Related projects

- [pHierCC](https://github.com/zheminzhou/pHierCC) -- Original HierCC implementation by Zhou et al.
- [plepiseq-wgs-pipeline](https://github.com/BioinfoPZH/plepiseq-wgs-pipeline) -- WGS analysis pipeline (consumer of clustering results)
- [plepiseq-phylogenetic-pipeline](https://github.com/mkadlof/plepiseq-phylogenetic-pipeline) -- Phylogenetic analysis pipeline (uses HierCC cluster assignments)

---

## Citation

If you use this software or the clustering results it produces, please cite the original pHierCC publication:

> Zhou Z, Charlesworth J, Achtman M (2020). HierCC: A multi-level clustering scheme for population assignments based on core genome MLST. *Bioinformatics*, 37(19), 3149-3155. DOI: [10.1093/bioinformatics/btab234](https://doi.org/10.1093/bioinformatics/btab234)

---

## License

This project is licensed under the **GPL-3.0 License** (same as the original pHierCC). See the [LICENSE](LICENSE) file for details.
