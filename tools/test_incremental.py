#!/usr/bin/env python3
"""
Lightweight verification of incremental condensed-vector expansion logic.
Tests only the index math and copy operations -- does NOT require numba
or SharedArray (those run inside the Docker container).

We simulate the scenario with plain numpy:
  - Build a full 9x9 condensed distance vector (random but deterministic)
  - Expand to 19x19 by copying old distances and filling new pairs
  - Build a full 19x19 from scratch
  - Compare the "old-old" pairs to verify the copy was correct
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(format="%(asctime)s | %(message)s", stream=sys.stdout, level=logging.INFO)

BASE = os.path.dirname(os.path.abspath(__file__))
PROFILE_9 = os.path.join(BASE, "test_data", "profiles_9ST.list")
PROFILE_19 = os.path.join(BASE, "test_data", "profiles_19ST.list")


def prepare_mat(profile_file):
    mat = pd.read_csv(profile_file, sep="\t", header=None, dtype=str).values
    allele_columns = np.array([i == 0 or (not h.startswith("#")) for i, h in enumerate(mat[0])])
    mat = mat[1:, allele_columns]
    try:
        mat = mat.astype(np.int32)
        mat = mat[mat.T[0] > 0]
        names = mat.T[0].copy()
    except Exception:
        names = mat.T[0].copy()
        mat.T[0] = np.arange(1, mat.shape[0] + 1)
        mat = mat.astype(np.int32)
    mat[mat < 0] = 0
    return mat, names


def condensed_index(n, i, j):
    """Position of pair (i, j) [i < j] in a length-n*(n-1)/2 condensed vector."""
    return n * i - i * (i + 1) // 2 + (j - i - 1)


def compute_dual_dist_pair(mat, i, j, allowed_missing=0.05):
    """Pure-python version of dual_dist_squareform distance for a single pair."""
    n_loci = mat.shape[1]
    ql = np.sum(mat[i] > 0)
    rl, ad, al = 0.0, 1e-4, 1e-4
    for k in range(n_loci):
        if mat[j, k] > 0:
            rl += 1
            if mat[i, k] > 0:
                al += 1
                if mat[i, k] != mat[j, k]:
                    ad += 1
    ll = max(ql, rl) - allowed_missing * n_loci
    if ll > al:
        ad += ll - al
        al = ll
    return int(ad / al * n_loci + 0.5)


def compute_full_condensed(mat, allowed_missing=0.05):
    """Compute the full condensed distance vector (pure Python, no numba)."""
    n = mat.shape[0]
    size = n * (n - 1) // 2
    dist = np.zeros(size, dtype=np.int16)
    alleles = mat[:, 1:]  # strip ST ID column
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_dual_dist_pair(alleles, i, j, allowed_missing)
            dist[condensed_index(n, i, j)] = np.int16(d)
    return dist


def compute_full_dist1(mat, allowed_missing=0.05):
    """Compute the full lower-triangular distance matrix (depth=1), pure Python."""
    n = mat.shape[0]
    alleles = mat[:, 1:]
    n_loci = alleles.shape[1]
    dist = np.zeros((n, n, 1), dtype=np.int16)
    for i in range(n):
        ql = np.sum(alleles[i] > 0)
        for j in range(i):
            rl, ad, al = 0.0, 1e-4, 1e-4
            for k in range(n_loci):
                if alleles[j, k] > 0:
                    rl += 1
                    if alleles[i, k] > 0:
                        al += 1
                        if alleles[i, k] != alleles[j, k]:
                            ad += 1
            ll2 = ql - allowed_missing * n_loci
            if ll2 > al:
                ad += ll2 - al
                al = ll2
            dist[i, j, 0] = int(ad / al * n_loci + 0.5)
    return dist


def copy_old_condensed(old_dist, old_n, n_new):
    """
    Copy old condensed vector into the correct positions of a new larger one.
    This is the core logic from ExpandSquareform.
    """
    new_dist = np.zeros(n_new * (n_new - 1) // 2, dtype=np.int16)
    for i in range(old_n):
        old_start = old_n * i - i * (i + 1) // 2
        new_start = n_new * i - i * (i + 1) // 2
        length = old_n - 1 - i
        if length > 0:
            new_dist[new_start : new_start + length] = old_dist[old_start : old_start + length]
    return new_dist


def test_condensed_copy():
    """Test that the row-by-row copy puts old distances in the right places."""
    logging.info("--- Test 1: Condensed vector copy ---")
    mat9, _ = prepare_mat(PROFILE_9)
    mat19, _ = prepare_mat(PROFILE_19)

    absence9 = np.sum(mat9 <= 0, 1)
    mat9[:] = mat9[np.argsort(absence9, kind="mergesort")]
    old_ordering = mat9.T[0].copy()
    old_n = len(old_ordering)

    # Build old condensed
    dist0_old = compute_full_condensed(mat9)

    # Reorder mat19: old STs in old order, new STs sorted by missing alleles
    old_id_to_pos = {int(st): pos for pos, st in enumerate(old_ordering)}
    old_mask = np.array([int(row[0]) in old_id_to_pos for row in mat19])
    new_mask = ~old_mask

    old_rows = mat19[old_mask]
    old_sort = np.array([old_id_to_pos[int(st)] for st in old_rows.T[0]])
    old_rows = old_rows[np.argsort(old_sort)]

    new_rows = mat19[new_mask]
    new_absence = np.sum(new_rows[:, 1:] <= 0, axis=1)
    new_rows = new_rows[np.argsort(new_absence, kind="mergesort")]

    mat19_reordered = np.vstack([old_rows, new_rows])
    n_new = mat19_reordered.shape[0]

    logging.info(f"  Old ordering (9 STs): {old_ordering.tolist()}")
    logging.info(f"  New ordering (19 STs): {mat19_reordered.T[0].tolist()}")

    # Copy old distances into new condensed vector
    new_dist = copy_old_condensed(dist0_old, old_n, n_new)

    # Compute full 19-ST distances for the reordered matrix
    dist0_full = compute_full_condensed(mat19_reordered)

    # Verify: for all old-old pairs, copied values must match full computation
    mismatches = 0
    checked = 0
    for i in range(old_n):
        for j in range(i + 1, old_n):
            pos = condensed_index(n_new, i, j)
            checked += 1
            if new_dist[pos] != dist0_full[pos]:
                mismatches += 1
                logging.error(
                    f"  MISMATCH old-old pair ({i},{j}) pos={pos}: "
                    f"copied={new_dist[pos]} expected={dist0_full[pos]}"
                )

    logging.info(f"  Checked {checked} old-old pairs, mismatches: {mismatches}")

    # Now fill in the new pairs and verify full match
    for i in range(n_new):
        j_start = old_n if i < old_n else i + 1
        for j in range(j_start, n_new):
            pos = condensed_index(n_new, i, j)
            new_dist[pos] = dist0_full[pos]  # simulate computation

    total_mismatch = np.sum(new_dist != dist0_full)
    logging.info(f"  After filling new pairs, total mismatches: {total_mismatch}")
    assert mismatches == 0, "Old-old pair copy failed!"
    assert total_mismatch == 0, "Full vector does not match!"
    logging.info("  PASSED")


def test_dist1_copy():
    """Test that old dist1 block can be directly embedded in the new matrix."""
    logging.info("--- Test 2: dist1 matrix copy ---")
    mat9, _ = prepare_mat(PROFILE_9)
    mat19, _ = prepare_mat(PROFILE_19)

    absence9 = np.sum(mat9 <= 0, 1)
    mat9[:] = mat9[np.argsort(absence9, kind="mergesort")]
    old_ordering = mat9.T[0].copy()
    old_n = len(old_ordering)

    dist1_old = compute_full_dist1(mat9)

    # Reorder mat19
    old_id_to_pos = {int(st): pos for pos, st in enumerate(old_ordering)}
    old_mask = np.array([int(row[0]) in old_id_to_pos for row in mat19])
    new_mask = ~old_mask
    old_rows = mat19[old_mask]
    old_sort = np.array([old_id_to_pos[int(st)] for st in old_rows.T[0]])
    old_rows = old_rows[np.argsort(old_sort)]
    new_rows = mat19[new_mask]
    new_absence = np.sum(new_rows[:, 1:] <= 0, axis=1)
    new_rows = new_rows[np.argsort(new_absence, kind="mergesort")]
    mat19_reordered = np.vstack([old_rows, new_rows])

    dist1_full = compute_full_dist1(mat19_reordered)

    # The old dist1 block should be identical to the top-left of the full matrix
    mismatches = 0
    for i in range(old_n):
        for j in range(i):
            if dist1_old[i, j, 0] != dist1_full[i, j, 0]:
                mismatches += 1
                logging.error(
                    f"  MISMATCH dist1[{i},{j}]: old={dist1_old[i, j, 0]} "
                    f"full={dist1_full[i, j, 0]}"
                )

    logging.info(f"  Checked {old_n * (old_n - 1) // 2} old-old entries, mismatches: {mismatches}")
    assert mismatches == 0, "dist1 old block does not match!"
    logging.info("  PASSED")


def test_new_pair_positions():
    """Verify we compute the right positions for new pairs."""
    logging.info("--- Test 3: New-pair position coverage ---")
    old_n = 5
    n_new = 8
    total = n_new * (n_new - 1) // 2  # 28

    covered_old = set()
    covered_new = set()

    # Old-old pairs (copied)
    for i in range(old_n):
        for j in range(i + 1, old_n):
            covered_old.add(condensed_index(n_new, i, j))

    # New pairs (computed)
    for i in range(n_new):
        j_start = old_n if i < old_n else i + 1
        for j in range(j_start, n_new):
            covered_new.add(condensed_index(n_new, i, j))

    all_positions = set(range(total))
    covered = covered_old | covered_new
    uncovered = all_positions - covered
    overlap = covered_old & covered_new

    logging.info(
        f"  Total positions: {total}, old: {len(covered_old)}, "
        f"new: {len(covered_new)}, overlap: {len(overlap)}, "
        f"uncovered: {len(uncovered)}"
    )

    assert len(uncovered) == 0, f"Uncovered positions: {uncovered}"
    assert len(overlap) == 0, f"Overlapping positions: {overlap}"
    assert len(covered) == total
    logging.info("  PASSED")


def test_mixed_ids():
    """
    Test the scenario where public (numeric) and local (text) STs are combined.

    Week 1: 9 public STs (profiles_9ST.list)
    Week 2: 9 public + 9 local STs (concatenated)

    Verifies that names-based ordering correctly identifies old STs even when
    prepare_mat assigns synthetic sequential integers to the mixed-ID file.
    """
    logging.info("--- Test 4: Mixed numeric + text ST IDs ---")
    PROFILE_LOCAL = os.path.join(BASE, "test_data", "profiles_9ST_local.list")

    # Week 1: public only (numeric IDs)
    mat_pub, names_pub = prepare_mat(PROFILE_9)
    absence = np.sum(mat_pub <= 0, 1)
    sort_idx = np.argsort(absence, kind="mergesort")
    mat_pub[:] = mat_pub[sort_idx]
    ordered_names_w1 = [str(names_pub[i]) for i in sort_idx]
    old_n = len(ordered_names_w1)

    logging.info(f"  Week 1 names: {ordered_names_w1}")

    # Week 2: combine public + local into one file (simulated)
    mat_local, names_local = prepare_mat(PROFILE_LOCAL)

    # When files are concatenated, both go through prepare_mat together.
    # Simulate: create a combined profile file in memory
    import tempfile

    with open(PROFILE_9) as f:
        lines_pub = f.readlines()
    with open(PROFILE_LOCAL) as f:
        lines_local = f.readlines()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".list", delete=False, dir=BASE) as tmp:
        tmp.write(lines_pub[0])  # header
        for line in lines_pub[1:]:
            tmp.write(line)
        for line in lines_local[1:]:  # skip header
            tmp.write(line)
        tmp_path = tmp.name

    try:
        mat_combined, names_combined = prepare_mat(tmp_path)
    finally:
        os.unlink(tmp_path)

    logging.info(f"  Combined names: {[str(n) for n in names_combined]}")
    logging.info(f"  Combined mat.T[0]: {mat_combined.T[0].tolist()}")

    # names_combined should have original IDs: ["1","2",...,"9","local_1",...,"local_9"]
    # mat_combined.T[0] will be [1,2,...,18] (sequential, because text IDs forced except branch)
    combined_name_set = set(str(n) for n in names_combined)

    # Verify all week-1 names are found in the combined set
    missing = [n for n in ordered_names_w1 if n not in combined_name_set]
    assert len(missing) == 0, f"Week 1 names missing from combined: {missing}"

    # Build the matid→name mapping (as the real code does)
    matid_to_name = {
        int(mat_combined[i, 0]): str(names_combined[i]) for i in range(mat_combined.shape[0])
    }
    name_for_row = [matid_to_name[int(row[0])] for row in mat_combined]

    # Simulate incremental reordering
    old_name_to_pos = {st: pos for pos, st in enumerate(ordered_names_w1)}
    old_mask = np.array([n in old_name_to_pos for n in name_for_row])
    new_mask = ~old_mask

    n_old_found = old_mask.sum()
    n_new_found = new_mask.sum()

    logging.info(f"  Old STs matched: {n_old_found}, new STs: {n_new_found}")
    assert n_old_found == old_n, f"Expected {old_n} old STs, found {n_old_found}"
    assert n_new_found == len(names_combined) - old_n

    # Verify the old rows get sorted back to their original order
    old_rows = mat_combined[old_mask]
    old_names_subset = [n for n, m in zip(name_for_row, old_mask) if m]
    old_sort = np.array([old_name_to_pos[n] for n in old_names_subset])
    old_rows_sorted = old_rows[np.argsort(old_sort)]

    # The allele data for each old ST should match between week1 and combined
    for i in range(old_n):
        w1_alleles = mat_pub[i, 1:]
        combined_alleles = old_rows_sorted[i, 1:]
        assert np.array_equal(w1_alleles, combined_alleles), (
            f"Allele mismatch for old ST at position {i}"
        )

    logging.info("  PASSED")


def test_local_pushed_down():
    """
    Test that local_ STs are always placed after public STs in the ordering,
    both for full-mode and incremental-mode sorting.
    """
    logging.info("--- Test 5: local_ STs pushed to bottom ---")
    PROFILE_LOCAL = os.path.join(BASE, "test_data", "profiles_9ST_local.list")

    import tempfile

    with open(PROFILE_9) as f:
        lines_pub = f.readlines()
    with open(PROFILE_LOCAL) as f:
        lines_local = f.readlines()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".list", delete=False, dir=BASE) as tmp:
        tmp.write(lines_pub[0])
        for line in lines_pub[1:]:
            tmp.write(line)
        for line in lines_local[1:]:
            tmp.write(line)
        tmp_path = tmp.name

    try:
        mat, names = prepare_mat(tmp_path)
    finally:
        os.unlink(tmp_path)

    mat_names = [str(n) for n in names]

    # --- Full-mode sort: public first (by absence), local last (by absence) ---
    pub_idx = [i for i, n in enumerate(mat_names) if not n.startswith("local_")]
    loc_idx = [i for i, n in enumerate(mat_names) if n.startswith("local_")]

    pub_rows = mat[np.array(pub_idx)]
    pub_names = [mat_names[i] for i in pub_idx]
    pub_abs = np.sum(pub_rows[:, 1:] <= 0, axis=1)
    pub_order = np.argsort(pub_abs, kind="mergesort")
    pub_rows = pub_rows[pub_order]
    pub_names = [pub_names[i] for i in pub_order]

    loc_rows = mat[np.array(loc_idx)]
    loc_names = [mat_names[i] for i in loc_idx]
    loc_abs = np.sum(loc_rows[:, 1:] <= 0, axis=1)
    loc_order = np.argsort(loc_abs, kind="mergesort")
    loc_rows = loc_rows[loc_order]
    loc_names = [loc_names[i] for i in loc_order]

    ordered = pub_names + loc_names
    logging.info(f"  Full-mode ordering: {ordered}")

    split_point = len(pub_names)
    for i in range(split_point):
        assert not ordered[i].startswith("local_"), (
            f"Public section has local ST at position {i}: {ordered[i]}"
        )
    for i in range(split_point, len(ordered)):
        assert ordered[i].startswith("local_"), (
            f"Local section has public ST at position {i}: {ordered[i]}"
        )

    # --- Incremental mode: old order preserved, new STs split public/local ---
    old_ordered = ordered[:9]  # simulate week-1 = first 9 public STs
    logging.info(f"  Week-1 ordering (first 9 public): {old_ordered}")

    matid_to_name = {int(mat[i, 0]): str(names[i]) for i in range(mat.shape[0])}
    name_for_row = [matid_to_name[int(row[0])] for row in mat]

    old_name_to_pos = {st: pos for pos, st in enumerate(old_ordered)}
    old_mask = np.array([n in old_name_to_pos for n in name_for_row])
    new_mask = ~old_mask

    new_rows = mat[new_mask]
    new_names_subset = [n for n, m in zip(name_for_row, new_mask) if m]

    new_pub_idx = [i for i, n in enumerate(new_names_subset) if not n.startswith("local_")]
    new_loc_idx = [i for i, n in enumerate(new_names_subset) if n.startswith("local_")]

    if new_pub_idx:
        np_rows = new_rows[np.array(new_pub_idx)]
        np_names = [new_names_subset[i] for i in new_pub_idx]
        np_abs = np.sum(np_rows[:, 1:] <= 0, axis=1)
        np_order = np.argsort(np_abs, kind="mergesort")
        np_rows = np_rows[np_order]
        np_names = [np_names[i] for i in np_order]
    else:
        np_names = []

    if new_loc_idx:
        nl_rows = new_rows[np.array(new_loc_idx)]
        nl_names = [new_names_subset[i] for i in new_loc_idx]
        nl_abs = np.sum(nl_rows[:, 1:] <= 0, axis=1)
        nl_order = np.argsort(nl_abs, kind="mergesort")
        nl_names = [nl_names[i] for i in nl_order]
    else:
        nl_names = []

    incremental_order = old_ordered + np_names + nl_names
    logging.info(f"  Incremental ordering: {incremental_order}")

    # All local_ should come after all non-local
    first_local = None
    for i, n in enumerate(incremental_order):
        if n.startswith("local_"):
            first_local = i
            break
    if first_local is not None:
        for i in range(first_local, len(incremental_order)):
            assert incremental_order[i].startswith("local_"), (
                f"Non-local ST found after first local at pos {i}: {incremental_order[i]}"
            )

    logging.info("  PASSED")


def main():
    test_new_pair_positions()
    test_condensed_copy()
    test_dist1_copy()
    test_mixed_ids()
    test_local_pushed_down()
    logging.info("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
