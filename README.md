# plepiseq-cluster -- Hierarchical clustering of cgMLST profiles

This project is part of the PleEpiSeq project, co-funded by the European Union.

A modified version of [pHierCC](https://github.com/zheminzhou/pHierCC) (Zhou et al.) optimized for large-scale cgMLST hierarchical clustering (600,000+ sequence types, 3,002 loci) with significantly reduced RAM usage and faster distance computation.

---

## Table of contents

1. [Features](#features)
2. [Quick start](#quick-start)
3. [Requirements](#requirements)
4. [Running pHierCC](#running-phiercc)
5. [Clustering results](#clustering-results-github-releases)
6. [Repository structure](#repository-structure)
7. [Related projects](#related-projects)
8. [Citation](#citation)
9. [License](#license)

---

## Features

- **int16 distance matrices** -- Modified SciPy's `hierarchy.py` and `_hierarchy.pyx` to perform hierarchical clustering directly on `np.int16` distances, reducing RAM requirments 
- **Numba parallel threading** -- Replaced original parrarelization approach with Numba. This lowers memory overhead and imptove load balancing on triangular workloads.
- **Incremental distance computation** -- Add ability to reuse previously computed distance matrices to speed up calculation that include novel STs reducing calculation time.
- **Mixed ST identifiers** -- Support for both numeric and text-based sequence type identifiers.
- **Dockerized build** --  a single Docker image to run all calculations.

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
| `--clustering_method` | Linkage criterion: `single` or `complete` | `single` |
| `-m`, `--allowed_missing` | Allowed proportion of missing genes in pairwise comparisons | 0.05 |
| `-a`, `--profile_distance0` | Pre-computed condensed distance matrix (`.npy`) | *auto-detected* |
| `-b`, `--profile_distance1` | Pre-computed full distance matrix (`.npy`) | *auto-detected* |
| `--clean` | Force full recalculation, removing cached distance matrices | `false` |

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

### Complete linkage (reusing distance matrices)

Complete linkage clustering can reuse distance matrices computed during a single linkage run, avoiding duplicate calculation:

```bash
docker run --rm \
    --volume /path/to/workdir:/dane:rw \
    --user $(id -u):$(id -g) \
    --ulimit nofile=262144:262144 \
    plepiseq-cluster:latest \
    --profile /dane/profiles.list.gz \
    --profile_distance0 /dane/dist0.npy \
    --profile_distance1 /dane/dist1.npy \
    -n 1 --clustering_method complete
```

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
├── Dockerfile                      # Docker image build (custom SciPy + Numba + TBB)
├── src/
│   ├── pHierCC.py                  # Main clustering script (CLI entrypoint)
│   └── getDistance.py              # Numba parallel distance computation kernels
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
