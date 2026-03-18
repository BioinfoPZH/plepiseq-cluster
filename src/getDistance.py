import logging

import numba as nb
import numpy as np

# ---------------------------------------------------------------------------
# Numba-parallel distance computation using prange + TBB work-stealing.
# No multiprocessing Pool, no SharedArray.
# ---------------------------------------------------------------------------


@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _squareform_numba_parallel(mat, allowed_missing=0.05):
    """Condensed (squareform) pairwise distance – all pairs."""
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql_arr[i] = c

    size = n * (n - 1) // 2
    dist = np.zeros(size, dtype=np.int16)

    for i in nb.prange(n):
        qi = np.float64(ql_arr[i])
        for j in range(i + 1, n):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4
            ll = max(qi, np.float64(ql_arr[j])) - allowed
            if ll > al:
                ad += ll - al
                al = ll
            pos = n * i - i * (i + 1) // 2 + (j - i - 1)
            dist[pos] = np.int16(ad / al * n_loci + 0.5)

    return dist


@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _dist1_numba_parallel(mat, start=0, allowed_missing=0.05, depth=0):
    """Full lower-triangular distance matrix – rows [start, n)."""
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql_arr[i] = c

    dist = np.zeros((n - start, n, 1), dtype=np.int16)

    for i in nb.prange(start, n):
        qi = np.float64(ql_arr[i])
        for j in range(i):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4

            if depth == 1:
                ll2 = qi - allowed
                if ll2 > al:
                    ad += ll2 - al
                    al = ll2
                dist[i - start, j, 0] = np.int16(ad / al * n_loci + 0.5)

            if depth == 0:
                ll = max(qi, np.float64(ql_arr[j])) - allowed
                if ll > al:
                    ad += ll - al
                    al = ll
                dist[i - start, j, 0] = np.int16(ad / al * n_loci + 0.5)

    return dist


@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _squareform_append_numba_parallel(mat, n_old, dist, allowed_missing=0.05):
    """Compute only new-pair distances (prange), write into pre-allocated dist."""
    n = mat.shape[0]
    n_loci = mat.shape[1]
    allowed = allowed_missing * n_loci

    ql_arr = np.empty(n, dtype=np.int32)
    for i in nb.prange(n):
        c = np.int32(0)
        for k in range(n_loci):
            c += np.int32(mat[i, k] > 0)
        ql_arr[i] = c

    for i in nb.prange(n):
        qi = np.float64(ql_arr[i])
        j_start = n_old if i < n_old else i + 1
        for j in range(j_start, n):
            al_int = np.int32(0)
            ad_int = np.int32(0)
            for k in range(n_loci):
                vi = mat[i, k]
                vj = mat[j, k]
                both = np.int32(vi > 0) & np.int32(vj > 0)
                al_int += both
                ad_int += both & np.int32(vi != vj)

            ad = np.float64(ad_int) + 1e-4
            al = np.float64(al_int) + 1e-4
            ll = max(qi, np.float64(ql_arr[j])) - allowed
            if ll > al:
                ad += ll - al
                al = ll
            pos = n * i - i * (i + 1) // 2 + (j - i - 1)
            dist[pos] = np.int16(ad / al * n_loci + 0.5)


# ---------------------------------------------------------------------------
# Public API called from pHierCC
# ---------------------------------------------------------------------------


def GetSquareformParallel(data, n_threads, allowed_missing=0.0):
    """Compute condensed distance matrix (dist0)."""
    nb.set_num_threads(n_threads)
    logging.info(f"Numba parallel: using {nb.get_num_threads()} threads")

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _squareform_numba_parallel(warmup, 0.05)

    dist = _squareform_numba_parallel(data[:, 1:], allowed_missing)
    return dist


def GetDistanceParallel(data, n_threads, start=0, allowed_missing=0.0, depth=0):
    """Compute full lower-triangular distance matrix (dist1)."""
    nb.set_num_threads(n_threads)

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _dist1_numba_parallel(warmup, 0, 0.05, depth)

    dist = _dist1_numba_parallel(data[:, 1:], start, allowed_missing, depth)
    return dist


def ExpandSquareformParallel(old_dist_path, old_n, new_mat, n_threads, allowed_missing=0.0):
    """Expand condensed distance vector with newly appended STs (dist0 incremental)."""
    nb.set_num_threads(n_threads)
    n_new = new_mat.shape[0]
    old_dist = np.load(old_dist_path, mmap_mode="r", allow_pickle=True)

    new_size = int(n_new * (n_new - 1) / 2)
    dist = np.zeros(new_size, dtype=np.int16)

    logging.info(f"Copying old condensed distances ({old_n} STs) into new vector ({n_new} STs)")
    for i in range(old_n):
        old_start = old_n * i - i * (i + 1) // 2
        new_start = n_new * i - i * (i + 1) // 2
        length = old_n - 1 - i
        if length > 0:
            dist[new_start : new_start + length] = old_dist[old_start : old_start + length]
    del old_dist

    n_new_sts = n_new - old_n
    total_new = old_n * n_new_sts + n_new_sts * (n_new_sts - 1) // 2
    logging.info(f"Computing {total_new} new pairwise distances ({n_new_sts} new STs)")

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    warmup_d = np.zeros(int(4 * 3 / 2), dtype=np.int16)
    _squareform_append_numba_parallel(warmup, 2, warmup_d, 0.05)

    _squareform_append_numba_parallel(new_mat[:, 1:], old_n, dist, allowed_missing)
    return dist


def ExpandDistanceParallel(old_dist_path, old_n, new_mat, n_threads, allowed_missing=0.0, depth=0):
    """Expand full distance matrix with newly appended STs (dist1 incremental)."""
    nb.set_num_threads(n_threads)
    n_new = new_mat.shape[0]

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _dist1_numba_parallel(warmup, 0, 0.05, depth)

    new_rows = _dist1_numba_parallel(new_mat[:, 1:], old_n, allowed_missing, depth)

    full_dist = np.zeros((n_new, n_new, 1), dtype=np.int16)
    old_dist = np.load(old_dist_path, mmap_mode="r", allow_pickle=True)
    full_dist[:old_n, :old_n, :] = old_dist[:, :, :]
    del old_dist

    full_dist[old_n:, :, :] = new_rows
    del new_rows

    return full_dist
