# Changelog
All notable changes to this project will be documented in this file.

## [0.2.4] - 2026-03-15
### Changed
- Merged `plepiseq_bin/` into `tools/`; all scripts now live under a single directory.
- `download_profile_Campylo.py` now accepts `-o`/`--output` to write directly to the target path; removed `mv` workaround in `run_clustering.sh`.
- Added HTTP error handling and missing-scheme check to `download_profile_Campylo.py`.

## [0.2.3] - 2026-03-15
### Changed
- Moved core scripts to `src/` and dropped the `_github` suffix (`pHierCC_github.py` → `src/pHierCC.py`, `getDistance_github.py` → `src/getDistance.py`).
- Moved utility scripts to `tools/` (`compare_hiercc.py`, `test_incremental.py`).
- Updated Dockerfile `COPY` paths and README repository structure accordingly.

## [0.2.2] - 2026-03-15
### Changed
- Renamed `cluster/` to `scipy_patches/`, keeping only the two modified files (`hierarchy.py`, `_hierarchy.pyx`); removed 13 unmodified SciPy files.
- Updated Dockerfile `COPY` paths accordingly.

## [0.2.1] - 2026-03-15
### Changed
- Replaced plaintext `README` with comprehensive `README.md` following plepiseq project conventions (features, quick start, CLI reference, repository structure, related projects, citation, license).

## [0.2.0] - 2026-03-14
### Changed
- Rewritten `tools/run_clustering.sh` (formerly `plepiseq_bin/run_clustering.sh`) to support incremental distance matrix computation by preserving `.npy` artefacts between weekly runs.
- Added `--clean` flag to the wrapper script, passed through to pHierCC to force full recalculation.
- Replaced `git add/commit/push` of clustering results with `gh release create`, publishing output files as GitHub Release assets instead of committing binary data to the repository.
- Removed `plepiseq_data/` from git tracking and purged historical binary blobs (reduced repository size from ~2 GiB to ~120 KiB).
- Added `set -euo pipefail` to the wrapper script for fail-fast behaviour.
- Added `gh` CLI availability check at script startup.

## [0.1.0] - 2026-03-14
### Added
- Initial working version based on the original [pHierCC](https://github.com/zheminzhou/pHierCC) by Zhou et al.
- Modified SciPy's `hierarchy.py` and `_hierarchy.pyx` to accept `np.int16` distance matrices, reducing RAM usage from `float64` (~8x saving) during hierarchical clustering.
- Replaced multiprocessing Pool + SharedArray distance computation with Numba `prange` thread parallelism and TBB work-stealing scheduler.
- Incremental distance matrix expansion: reuse previous run's `dist0.npy`, `dist1.npy`, and `ordering.npy` to avoid full recalculation when new STs are appended.
- Support for mixed numeric and text-based ST identifiers (e.g. public + `local_` profiles), with local STs always sorted to the bottom of the distance matrix.
- `--clean` flag to force full recalculation even when previous run artefacts exist.
- Dockerized build with custom SciPy compilation for `int16` clustering support.
- Weekly clustering wrapper script (`tools/run_clustering.sh`) for Salmonella, Escherichia, and Campylobacter.
