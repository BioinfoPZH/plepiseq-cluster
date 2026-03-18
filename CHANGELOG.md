# Changelog
All notable changes to this project will be documented in this file.

## [0.2.8] - 2026-03-18
### Changed
- Add Ruff linting with GitHub Actions workflow and local pre-commit hook

## [0.2.7] - 2026-03-17
### Removed
- Dropped `-a`/`--profile_distance0` and `-b`/`--profile_distance1` CLI options. Distance matrices are now always stored as `dist0.npy` and `dist1.npy` in the profile directory. Incremental mode triggers only when all three artefacts (`dist0.npy`, `dist1.npy`, `ordering.npy`) are present.

## [0.2.6] - 2026-03-17
### Changed
- `--clustering_method` now accepts multiple values (e.g. `--clustering_method single --clustering_method complete`) to run both linkage methods in a single invocation, avoiding hours-long distance matrix reloads.
- `run_clustering.sh` reduced from two Docker invocations per species to one.
- Updated README documentation with multi-method usage examples.

## [0.2.5] - 2026-03-17
### Changed
- pHierCC now skips computation entirely (exit code 42) when the profile contains the same STs as the previous run, with set-level verification to catch swapped STs.
- `run_clustering.sh` captures per-species exit codes; skips complete linkage when single linkage reports no changes; only creates a GitHub Release when at least one species was updated.
- Refactored the three per-species clustering blocks in `run_clustering.sh` into a single loop.

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
