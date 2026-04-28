# Changelog
All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-04-21
Critical data-correctness release: the handling of mixed numeric + `local_*`
profiles introduced in 0.3.x silently corrupted both the `#ST_id` column and
the HC clustering cells for every real-world input. Any `.HierCC.gz` produced
by 0.3.2 through 0.3.4 must be regenerated. Pure-numeric, pre-sorted,
uniform-missing profiles (the pHierCC upstream test shape) are byte-identical
to 0.3.1.

### Fixed
- **Data-correctness regression (critical), bug B.** `src/pHierCC.py` write loop indexed into `names` (input order) instead of `ordered_names` (post-`_split_local` order) when applying the final `argsort` permutation. For any profile where `_split_local` actually reordered rows -- i.e. any real dataset whose rows don't all have the same number of missing alleles, including every production run that merges public STs with `local_*` STs -- this silently wrote the right HC clustering data under the wrong `#ST_id` label. Outputs from 0.3.2, 0.3.3 and 0.3.4 should be discarded and regenerated (run `tools/run_clustering.sh --clean` or delete `ordering.npy` to force a full recompute). Pure-numeric, pre-sorted, uniform-missing profiles were unaffected, which is why the bug escaped review in the 0.3.2 reproducer. See issue #8, bug 3.
- **Data-correctness regression (critical), bug D.** `src/getDistance.py::prepare_mat_streaming` flipped the whole file into "synthetic row-position" mode whenever any single row failed `int()` (e.g. the first `local_*` row), replacing every real ST id in `mat[:,0]` with `1..N`. Downstream, HC cells of numeric rows then referenced row positions instead of real ST ids (e.g. `local_1` appeared as `83780` in Campylobacter, and renaming a `local_*` row to any non-existing integer id made HC0 values jump around arbitrarily). `prepare_mat_streaming` now keeps real numeric ids for numeric rows and assigns `LOCAL_OFFSET + N` (currently `1_000_000 + N`) to `local_N` rows; the pHierCC write loop renders any HC cell `>= LOCAL_OFFSET` back to its original `local_N` string via the `matid_to_name` map. Anything else in column 0 (e.g. a stray header row, empty id, or an unknown non-numeric token) is now a hard error instead of silently corrupting all ids.
- `tools/run_clustering.sh`: `prepare_profile` replaced the positional `tail -n +2` strip with an `awk '$1 != hdr1'` filter on both the external and the local stream, and added a post-merge assertion that the final `profiles.list[.gz]` contains exactly one header row. A duplicate header row was observed in one Campylobacter run (Apr 2026) despite clean staged inputs; the origin could not be reliably reproduced (transient PubMLST response, mid-read overwrite, or a pre-`read_header_line` shell path are all plausible), so the merge is now tolerant of duplicate headers in either input and fails loudly if somehow more than one header slips through.

### Added
- `test_data/profiles_with_local_ids.list` and `test_data/profiles_with_local_ids_switched.list` -- a paired reproducer (the second file is the first with the `local_*` rows renamed to integer ids) used to confirm, after the fix, that the numeric block of the mixed run is byte-identical to the pure-numeric run and that `local_*` rows render as `local_N` strings throughout.

## [0.3.4] - 2026-04-21
### Fixed
- `tools/run_clustering.sh`: reading the external profile header via `zcat | head -n 1` under `set -euo pipefail` caused `zcat` to receive `SIGPIPE` on large inputs, which `pipefail` propagated and `set -e` turned into a silent abort inside the `$(…)` substitution (the script terminated at the first species). Replaced with a `read_header_line` helper that drains the remainder of the stream to `/dev/null`.

## [0.3.3] - 2026-04-18
### Added
- `tools/run_clustering.sh`: three optional per-species local-profile flags (`--salmonella-local`, `--escherichia-local`, `--campylobacter-local`). Each accepts a tab-separated plain-text profile with a header matching the external download and rows whose first column matches `^local_[0-9]+$`.
- `tools/run_clustering.sh`: `retry_download` helper retries each external download once after 300 s before aborting the whole run with exit code 1.
- `tools/run_clustering.sh`: merge step that writes a segmented `profiles.list[.gz]` (external numeric STs ascending, then `local_*` rows ascending by suffix). Raw download is staged as `profiles_external.list[.gz]` and the optional local copy as `profiles_local.list`; both are cleaned up on re-run and on `--clean`. Prevents issue #8 bug 2 by construction.
- `src/pHierCC.py`: non-fatal `logging.warning` if the loaded profile violates the segmented layout (numeric STs after `local_*` STs, or a non-ascending numeric block). Purely diagnostic; behaviour unchanged.

## [0.3.2] - 2026-04-18
### Fixed
- `.HierCC.gz` write loop now applies the same permutation to `names` that it applies to `res` via `np.argsort(res.T[0])`. Previously, for profiles not already sorted ascending by numeric ST id, the `#ST_id` column was written in input order while the HC columns were written in ST-id-ascending order, silently misaligning labels and clustering data. Output is byte-identical for pre-sorted inputs (the production case). See issue #8, bug 1 (inherited from upstream pHierCC).

## [0.3.1] - 2026-03-13
### Changed
- Extend `.HierCC.index` format with a dense `local_*` segment and an `__LOCAL_START__` sentinel separator. Pure-numeric files gain a trailing sentinel at EOF; the existing sparse numeric checkpoints are unchanged.

## [0.3.0] - 2026-03-18
### Added
- Multi-GPU CUDA acceleration for distance matrix computation
- Memory-efficient streaming profile loader replaces the pandas-based function

### Changed
- `getDistance.py` now contains both CPU (Numba prange) and GPU (CUDA) distance kernels, tile orchestrator, and profile parsing utilities.
- GPU mode always performs a full recalculation (incremental mode is disabled when `--gpu-ids` is specified).
- Removed `pandas` dependency from `pHierCC.py`.

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
