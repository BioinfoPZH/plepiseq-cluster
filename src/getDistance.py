"""
Distance computation for pHierCC -- CPU (Numba prange) and GPU (CUDA) paths.

CPU functions use Numba thread-level parallelism with TBB work-stealing.
GPU functions tile the pairwise computation across one or more CUDA devices.
"""

import gzip
import logging
import math
import multiprocessing as mp
import os
import queue
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass

import numba as nb
import numpy as np

try:
    from numba import cuda

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


# ---------------------------------------------------------------------------
# CPU Numba-parallel distance kernels
# ---------------------------------------------------------------------------


@nb.jit(nopython=True, parallel=True, fastmath=True, boundscheck=False)
def _squareform_numba_parallel(mat, allowed_missing=0.05):
    """Condensed (squareform) pairwise distance -- all pairs."""
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
    """Full lower-triangular distance matrix -- rows [start, n)."""
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
# CUDA tile kernels (only defined when numba.cuda is available)
# ---------------------------------------------------------------------------

if HAS_CUDA:

    @cuda.jit
    def _dist0_tile_kernel(a, b, qa, qb, out, n_loci, allowed_missing, same_block):
        """Condensed dist0 tile -- upper triangle (j > i) on diagonal."""
        i, j = cuda.grid(2)
        rows_a = out.shape[0]
        rows_b = out.shape[1]
        if i >= rows_a or j >= rows_b:
            return
        if same_block == 1 and j <= i:
            return

        qi = qa[i]
        qj = qb[j]
        al_int = 0
        ad_int = 0
        for k in range(n_loci):
            vi = a[i, k]
            vj = b[j, k]
            if vi > 0 and vj > 0:
                al_int += 1
                if vi != vj:
                    ad_int += 1

        ad = float(ad_int) + 1e-4
        al = float(al_int) + 1e-4
        ll = float(qi if qi >= qj else qj) - allowed_missing * n_loci
        if ll > al:
            ad += ll - al
            al = ll
        out[i, j] = np.int16(ad / al * n_loci + 0.5)

    @cuda.jit
    def _dist1_tile_kernel(a, b, qa, qb, out, n_loci, allowed_missing, same_block):
        """Full dist1 tile -- lower triangle (j < i) on diagonal.

        Uses depth=1 formula: penalty based on qi only (not max).
        """
        i, j = cuda.grid(2)
        rows_a = out.shape[0]
        rows_b = out.shape[1]
        if i >= rows_a or j >= rows_b:
            return
        if same_block == 1 and j >= i:
            return

        qi = qa[i]
        al_int = 0
        ad_int = 0
        for k in range(n_loci):
            vi = a[i, k]
            vj = b[j, k]
            if vi > 0 and vj > 0:
                al_int += 1
                if vi != vj:
                    ad_int += 1

        ad = float(ad_int) + 1e-4
        al = float(al_int) + 1e-4
        ll = float(qi) - allowed_missing * n_loci
        if ll > al:
            ad += ll - al
            al = ll
        out[i, j] = np.int16(ad / al * n_loci + 0.5)


# ---------------------------------------------------------------------------
# CUDA tile helpers (pure Python, no CUDA dependency)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TileJob:
    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def work(self) -> int:
        rows = self.i1 - self.i0
        cols = self.j1 - self.j0
        if self.i0 == self.j0:
            return rows * (rows - 1) // 2
        return rows * cols


def _condensed_index(i: int, j: int, n: int) -> int:
    """Index into the condensed upper-triangle vector for pair (i, j)."""
    return n * i - i * (i + 1) // 2 + (j - i - 1)


def _write_tile_condensed(mm: np.ndarray, tile: np.ndarray, i0: int, j0: int, n: int) -> None:
    """Write a dist0 tile into the condensed memmap vector."""
    rows, cols = tile.shape
    if i0 == j0:
        for li in range(rows):
            gj = j0 + li + 1
            nvals = cols - (li + 1)
            if nvals <= 0:
                continue
            pos = _condensed_index(i0 + li, gj, n)
            mm[pos : pos + nvals] = tile[li, li + 1 :]
    else:
        for li in range(rows):
            pos = _condensed_index(i0 + li, j0, n)
            mm[pos : pos + cols] = tile[li, :]


def _write_tile_full(mm: np.ndarray, tile: np.ndarray, i0: int, j0: int, n: int) -> None:
    """Write a dist1 tile into the full (n, n, 1) memmap."""
    rows, cols = tile.shape
    if i0 == j0:
        for li in range(1, rows):
            mm[i0 + li, j0 : j0 + li, 0] = tile[li, :li]
    else:
        for li in range(rows):
            mm[i0 + li, j0 : j0 + cols, 0] = tile[li, :]


def _build_tile_jobs(n: int, block_size: int, upper: bool = True) -> list[_TileJob]:
    """Build tile jobs covering the upper or lower triangle."""
    jobs: list[_TileJob] = []
    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        for j0 in range(0, n, block_size):
            j1 = min(j0 + block_size, n)
            if upper and j0 < i0:
                continue
            if not upper and j0 > i0:
                continue
            jobs.append(_TileJob(i0, i1, j0, j1))
    jobs.sort(key=lambda x: x.work, reverse=True)
    return jobs


def _save_temp_npy(path: str, arr: np.ndarray) -> None:
    """Write array as a flush-synced memmap .npy (for worker mmap access)."""
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    del mm


# ---------------------------------------------------------------------------
# CUDA multi-GPU orchestrator (only defined when numba.cuda is available)
# ---------------------------------------------------------------------------

if HAS_CUDA:

    def _warmup_cuda_kernel(kernel) -> None:
        """JIT-compile a tile kernel with a small dummy invocation."""
        warm = 64
        a = np.ones((warm, warm), dtype=np.int32)
        q = np.full(warm, warm, dtype=np.int32)
        d_a = cuda.to_device(a)
        d_q = cuda.to_device(q)
        d_out = cuda.device_array((warm, warm), dtype=np.int16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel[(4, 4), (16, 16)](d_a, d_a, d_q, d_q, d_out, warm, 0.05, 0)
        cuda.synchronize()

    def _gpu_worker(
        gpu_id,
        loci_path,
        ql_path,
        output_path,
        jobs_queue,
        result_queue,
        allowed_missing,
        threads_x,
        threads_y,
        mode,
        n,
    ):
        """Worker process: select GPU, process tiles, write results."""
        try:
            logging.getLogger("numba").setLevel(logging.WARNING)
            cuda.select_device(gpu_id)
            kernel = _dist0_tile_kernel if mode == "condensed" else _dist1_tile_kernel
            _warmup_cuda_kernel(kernel)

            loci = np.load(loci_path, mmap_mode="r")
            ql = np.load(ql_path, mmap_mode="r")
            output_mm = np.lib.format.open_memmap(output_path, mode="r+")
            n_loci = loci.shape[1]
            write_fn = _write_tile_condensed if mode == "condensed" else _write_tile_full

            tpb = (threads_x, threads_y)
            last_a_key = last_b_key = None
            d_a = d_b = d_qa = d_qb = None
            a_rows = b_rows = 0

            while True:
                job = jobs_queue.get()
                if job is None:
                    break

                i0, i1, j0, j1 = job.i0, job.i1, job.j0, job.j1
                a_key = (i0, i1)
                b_key = (j0, j1)

                if a_key != last_a_key:
                    a_host = np.ascontiguousarray(loci[i0:i1], dtype=np.int32)
                    qa_host = np.ascontiguousarray(ql[i0:i1], dtype=np.int32)
                    d_a = cuda.to_device(a_host)
                    d_qa = cuda.to_device(qa_host)
                    a_rows = a_host.shape[0]
                    last_a_key = a_key

                if b_key != last_b_key:
                    b_host = np.ascontiguousarray(loci[j0:j1], dtype=np.int32)
                    qb_host = np.ascontiguousarray(ql[j0:j1], dtype=np.int32)
                    d_b = cuda.to_device(b_host)
                    d_qb = cuda.to_device(qb_host)
                    b_rows = b_host.shape[0]
                    last_b_key = b_key

                d_out = cuda.device_array((a_rows, b_rows), dtype=np.int16)
                bpg = (
                    math.ceil(a_rows / tpb[0]),
                    math.ceil(b_rows / tpb[1]),
                )
                same = 1 if i0 == j0 else 0
                kernel[bpg, tpb](
                    d_a,
                    d_b,
                    d_qa,
                    d_qb,
                    d_out,
                    n_loci,
                    allowed_missing,
                    same,
                )

                tile = d_out.copy_to_host()
                write_fn(output_mm, tile, i0, j0, n)
                result_queue.put(("DONE", gpu_id, job.work))

            output_mm.flush()
            result_queue.put(("EXIT", gpu_id))

        except Exception:
            result_queue.put(("ERROR", gpu_id, traceback.format_exc()))
            raise

    def _compute_cuda_mgpu(
        loci,
        ql,
        output_path,
        mode,
        gpu_ids,
        allowed_missing,
        block_size,
        threads,
    ):
        """Orchestrate multi-GPU tiled distance computation.

        mode: "condensed" for dist0, "full" for dist1.
        Writes result directly to output_path as a .npy memmap.
        """
        logging.getLogger("numba").setLevel(logging.WARNING)
        n = loci.shape[0]

        available = len(list(cuda.gpus))
        if available < 1:
            raise RuntimeError("No CUDA GPUs detected.")
        if max(gpu_ids) >= available:
            raise RuntimeError(
                f"Requested GPU ids {gpu_ids}, but only {available} device(s) visible."
            )

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)

        temp_dir = tempfile.mkdtemp(prefix="dist_cuda_", dir=out_dir)
        loci_path = os.path.join(temp_dir, "loci.npy")
        ql_path = os.path.join(temp_dir, "ql.npy")

        logging.info(f"CUDA: writing temp arrays to {temp_dir}")
        _save_temp_npy(loci_path, loci)
        _save_temp_npy(ql_path, ql)

        if mode == "condensed":
            out_shape = (n * (n - 1) // 2,)
        else:
            out_shape = (n, n, 1)

        out_mm = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int16, shape=out_shape)
        out_mm.flush()
        del out_mm

        upper = mode == "condensed"
        jobs = _build_tile_jobs(n, block_size, upper=upper)
        total_work = sum(j.work for j in jobs)

        logging.info(f"CUDA: {len(jobs)} tiles, block_size={block_size}, GPUs={list(gpu_ids)}")

        ctx = mp.get_context("spawn")
        jobs_q = ctx.Queue(maxsize=max(8, len(gpu_ids) * 2))
        result_q: mp.Queue = ctx.Queue()

        workers = []
        for gid in gpu_ids:
            p = ctx.Process(
                target=_gpu_worker,
                args=(
                    gid,
                    loci_path,
                    ql_path,
                    output_path,
                    jobs_q,
                    result_q,
                    allowed_missing,
                    threads[0],
                    threads[1],
                    mode,
                    n,
                ),
                daemon=False,
            )
            p.start()
            workers.append(p)

        for job in jobs:
            jobs_q.put(job)
        for _ in gpu_ids:
            jobs_q.put(None)

        done_jobs = 0
        done_work = 0
        exited = 0
        last_report = time.time()

        try:
            while exited < len(gpu_ids):
                try:
                    msg = result_q.get(timeout=30)
                except queue.Empty:
                    bad = [p for p in workers if not p.is_alive() and p.exitcode not in (0, None)]
                    if bad:
                        raise RuntimeError(
                            "GPU worker(s) died: "
                            + ", ".join(f"pid={p.pid}, rc={p.exitcode}" for p in bad)
                        )
                    continue

                tag = msg[0]
                if tag == "DONE":
                    _, _gid, work = msg
                    done_jobs += 1
                    done_work += work
                    now = time.time()
                    if now - last_report >= 5:
                        pct = 100.0 * done_work / max(total_work, 1)
                        logging.info(f"CUDA progress: {done_jobs}/{len(jobs)} tiles, {pct:.1f}%")
                        last_report = now
                elif tag == "EXIT":
                    exited += 1
                elif tag == "ERROR":
                    _, gid_err, tb = msg
                    raise RuntimeError(f"GPU {gid_err} failed:\n{tb}")
                else:
                    raise RuntimeError(f"Unknown message: {msg}")
        finally:
            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
                    p.join()

        for p in workers:
            if p.exitcode != 0:
                raise RuntimeError(f"Worker PID {p.pid} exited with code {p.exitcode}")

        pct = 100.0 * done_work / max(total_work, 1)
        logging.info(f"CUDA complete: {done_jobs}/{len(jobs)} tiles, {pct:.1f}%")

        try:
            os.remove(loci_path)
            os.remove(ql_path)
            os.rmdir(temp_dir)
        except OSError:
            logging.warning(f"Could not remove temp dir {temp_dir}")


# ---------------------------------------------------------------------------
# Profile parsing (shared by CPU and GPU paths)
# ---------------------------------------------------------------------------


def _open_text_auto(path: str):
    """Open a text file, auto-detecting gzip compression."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _split_fields(line: str) -> list[str]:
    line = line.rstrip("\r\n")
    fields = line.split("\t")
    if len(fields) == 1:
        fields = line.split()
    return fields


def prepare_mat_streaming(profile_file: str):
    """
    Memory-efficient profile loader (two-pass, no pandas).

    Matches prepare_mat() semantics: first row is header, columns
    whose header starts with '#' are omitted (except column 0),
    negative allele values become 0, rows with ST <= 0 are dropped
    when IDs are numeric.

    Returns (mat, names) where mat is int32 with ST-id in column 0.
    """
    with _open_text_auto(profile_file) as fh:
        try:
            header = _split_fields(next(fh))
        except StopIteration:
            raise ValueError(f"Empty profile file: {profile_file}")

        keep_cols = [0] + [i for i, h in enumerate(header[1:], start=1) if not h.startswith("#")]

        raw_names: list[str] = []
        st_ids: list[int] = []
        numeric_ids = True

        for line in fh:
            if not line.strip():
                continue
            fields = _split_fields(line)
            st = fields[0]
            raw_names.append(st)
            if numeric_ids:
                try:
                    st_ids.append(int(st))
                except ValueError:
                    numeric_ids = False

    n_total = len(raw_names)
    if n_total == 0:
        raise ValueError(f"No rows in profile: {profile_file}")

    if numeric_ids:
        keep_row = np.array([x > 0 for x in st_ids], dtype=bool)
        n_rows = int(keep_row.sum())
        names = [str(x) for x in np.array(st_ids, dtype=np.int64)[keep_row]]
    else:
        keep_row = np.ones(n_total, dtype=bool)
        n_rows = n_total
        names = raw_names

    n_cols = len(keep_cols)
    mat = np.empty((n_rows, n_cols), dtype=np.int32)

    with _open_text_auto(profile_file) as fh:
        next(fh)  # skip header
        src_row = -1
        dst_row = 0
        for line in fh:
            if not line.strip():
                continue
            src_row += 1
            if not keep_row[src_row]:
                continue
            fields = _split_fields(line)

            if numeric_ids:
                mat[dst_row, 0] = int(fields[0])
            else:
                mat[dst_row, 0] = dst_row + 1

            for dst_col, src_col in enumerate(keep_cols[1:], start=1):
                v = 0
                if src_col < len(fields):
                    try:
                        v = int(fields[src_col])
                    except ValueError:
                        v = 0
                if v < 0:
                    v = 0
                mat[dst_row, dst_col] = v
            dst_row += 1

    return mat, names


# ---------------------------------------------------------------------------
# CPU public API (called from pHierCC)
# ---------------------------------------------------------------------------


def GetSquareformParallel(data, n_threads, allowed_missing=0.0):
    """Compute condensed distance matrix (dist0) on CPU."""
    nb.set_num_threads(n_threads)
    logging.info(f"Numba parallel: using {nb.get_num_threads()} threads")

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _squareform_numba_parallel(warmup, 0.05)

    dist = _squareform_numba_parallel(data[:, 1:], allowed_missing)
    return dist


def GetDistanceParallel(data, n_threads, start=0, allowed_missing=0.0, depth=0):
    """Compute full lower-triangular distance matrix (dist1) on CPU."""
    nb.set_num_threads(n_threads)

    warmup = np.random.randint(0, 2, size=(4, 10)).astype(np.int32)
    _dist1_numba_parallel(warmup, 0, 0.05, depth)

    dist = _dist1_numba_parallel(data[:, 1:], start, allowed_missing, depth)
    return dist


def ExpandSquareformParallel(old_dist_path, old_n, new_mat, n_threads, allowed_missing=0.0):
    """Expand condensed distance vector with new STs (dist0 incremental)."""
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


def ExpandDistanceParallel(
    old_dist_path,
    old_n,
    new_mat,
    n_threads,
    allowed_missing=0.0,
    depth=0,
):
    """Expand full distance matrix with new STs (dist1 incremental)."""
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


# ---------------------------------------------------------------------------
# CUDA public API (called from pHierCC when --gpu-ids is specified)
# ---------------------------------------------------------------------------


def GetSquareformCUDA(
    mat,
    gpu_ids,
    allowed_missing,
    block_size,
    threads,
    output_path,
):
    """Compute condensed distance matrix (dist0) on GPU(s).

    Writes result directly to output_path as a .npy file and returns
    a read-only memmap of the result.
    """
    if not HAS_CUDA:
        raise RuntimeError(
            "numba.cuda is not available. Install numba-cuda or use the CPU path (omit --gpu-ids)."
        )

    loci = np.ascontiguousarray(mat[:, 1:], dtype=np.int32)
    ql = np.ascontiguousarray((loci > 0).sum(axis=1), dtype=np.int32)
    n = loci.shape[0]

    logging.info(f"CUDA dist0: {n} profiles, output length = {n * (n - 1) // 2}")
    _compute_cuda_mgpu(
        loci,
        ql,
        output_path,
        "condensed",
        gpu_ids,
        allowed_missing,
        block_size,
        threads,
    )
    return np.load(output_path, mmap_mode="r", allow_pickle=True)


def GetDistanceCUDA(
    mat,
    gpu_ids,
    allowed_missing,
    block_size,
    threads,
    output_path,
):
    """Compute full distance matrix (dist1, depth=1) on GPU(s).

    Writes result directly to output_path as a .npy file and returns
    a read-only memmap of the result.
    """
    if not HAS_CUDA:
        raise RuntimeError(
            "numba.cuda is not available. Install numba-cuda or use the CPU path (omit --gpu-ids)."
        )

    loci = np.ascontiguousarray(mat[:, 1:], dtype=np.int32)
    ql = np.ascontiguousarray((loci > 0).sum(axis=1), dtype=np.int32)
    n = loci.shape[0]

    logging.info(f"CUDA dist1: {n} profiles, output shape = ({n}, {n}, 1)")
    _compute_cuda_mgpu(
        loci,
        ql,
        output_path,
        "full",
        gpu_ids,
        allowed_missing,
        block_size,
        threads,
    )
    return np.load(output_path, mmap_mode="r", allow_pickle=True)
