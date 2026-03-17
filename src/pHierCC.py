#!/usr/bin/env python

#  pHierCC.py
#  pipeline for Hierarchical Clustering of cgMLST
#
#  Author: Zhemin Zhou
#  Lisence: GPLv3
#
#
#  Example of a rofile file (tab delimited):
#  ST_id gene1 gene2
#  1 1 1
#  2 1 2
#  To execute type:
#  pHierCC --profile <path> \
#          --cpus <int> \
#          --clustering_method single --clustering_method complete \
#          --allowed_missing <float> \
# Output files are named based on the linkage criterion used for clustering
# profile_{method}_linkage.HierCC.gz and profile_{method}_linkage.HierCC.index
# (one pair per requested method)


import sys, gzip, logging, click
import pandas as pd, numpy as np
from scipy.cluster.hierarchy import linkage
import os

try :
    from getDistance import GetSquareformParallel
    from getDistance import GetDistanceParallel
    from getDistance import ExpandSquareformParallel
    from getDistance import ExpandDistanceParallel
except :
    from .getDistance import GetSquareformParallel
    from .getDistance import GetDistanceParallel
    from .getDistance import ExpandSquareformParallel
    from .getDistance import ExpandDistanceParallel

logging.basicConfig(format='%(asctime)s | %(message)s', stream=sys.stdout, level=logging.INFO)


def prep_index(file_to_index, every=10000):
    """
    Text file with information where data regarding clustering of a given ST starts
    in a gz file. The index contain information about every 10,000 ST
    @param file_to_index:
    @type file_to_index:
    @param every:
    @type every:
    @return:
    @rtype:
    """
    if os.path.exists(file_to_index):
        output_file=file_to_index.replace(".gz", "")
        with open(f'{output_file}.index', 'w') as f, gzip.open(file_to_index) as f2:
            i = 0
            length = 0
            for line in f2:
                if (i % every) == 0:
                    elementy = list(map(lambda x: x.decode('utf-8', errors='replace'), line.split()))
                    f.write(f'{elementy[0]}\t{length}\n')
                length += len(line)
                i += 1
    else:
        raise Exception('Provided file does not exist')
    return True

def prepare_mat(profile_file):
    """
    Creates a numpy array from a profile file in tab-separated format. First row is
    assumed to contain information regarding allele names (columns where allele name starts with # are omitted). If ST
    identifiers are numeric the value that are lower than 0 are ignored.
    Allele variants with values lower than 0 are converted to 0.
    Profile should not contain non-int values (e.g. "-")

    :param profile_file: str, path to file containing a file with profile
    :return: a 2-element list, first element contains a matrix (values are stored as np.int32, second array
    stores names of sequence types (de facto first column from a profile file but without the first row)
    """
    mat = pd.read_csv(profile_file, sep='\t', header=None, dtype=str).values
    allele_columns = np.array([i == 0 or (not h.startswith('#')) for i, h in enumerate(mat[0])])
    mat = mat[1:, allele_columns]
    try:
        mat = mat.astype(np.int32)
        mat = mat[mat.T[0] > 0]
        names = mat.T[0].copy()
    except:
        names = mat.T[0].copy()
        mat.T[0] = np.arange(1, mat.shape[0]+1)
        mat = mat.astype(np.int32)
    mat[mat < 0] = 0
    return mat, names


def _split_local(rows, row_names):
    """
    Split rows into public and local STs.  Local STs (names starting with
    'local_') are always placed after public ones.  Within each group rows
    are sorted by number of missing alleles (ascending).

    Returns (pub_rows, loc_rows, pub_names_sorted, loc_names_sorted).
    """
    pub_idx = [i for i, n in enumerate(row_names) if not n.startswith('local_')]
    loc_idx = [i for i, n in enumerate(row_names) if n.startswith('local_')]

    if pub_idx:
        pub_rows = rows[np.array(pub_idx)]
        pub_names = [row_names[i] for i in pub_idx]
        pub_abs = np.sum(pub_rows[:, 1:] <= 0, axis=1)
        pub_order = np.argsort(pub_abs, kind='mergesort')
        pub_rows = pub_rows[pub_order]
        pub_names = [pub_names[i] for i in pub_order]
    else:
        pub_rows = np.empty((0, rows.shape[1]), dtype=rows.dtype)
        pub_names = []

    if loc_idx:
        loc_rows = rows[np.array(loc_idx)]
        loc_names = [row_names[i] for i in loc_idx]
        loc_abs = np.sum(loc_rows[:, 1:] <= 0, axis=1)
        loc_order = np.argsort(loc_abs, kind='mergesort')
        loc_rows = loc_rows[loc_order]
        loc_names = [loc_names[i] for i in loc_order]
    else:
        loc_rows = np.empty((0, rows.shape[1]), dtype=rows.dtype)
        loc_names = []

    return pub_rows, loc_rows, pub_names, loc_names


@click.command()
@click.option('-p', '--profile', help='[INPUT] name of a profile file consisting of a '
                                      'table of columns of the ST numbers and the allelic numbers, '
                                      'separated by tabs. Can be GZIPped.',
              required=True, type=click.Path())
@click.option('-m', '--allowed_missing', help='[INPUT; optional] Allowed proportion of '
                                              'missing genes in pairwise comparisons (Default: 0.05). ',
              default=0.05, type=float)
@click.option('-n', '--n_proc', help='[INPUT; optional] Number of processes (CPUs) to use '
                                     '(Default: 4).', default=4, type=int)
@click.option('--clustering_method', multiple=True,
              type=click.Choice(['single', 'complete']),
              default=('single',),
              help='Linkage criterion(s). Can be specified multiple times to '
                   'run several methods in a single invocation (e.g. '
                   '--clustering_method single --clustering_method complete).')
@click.option('--clean', is_flag=True, default=False,
              help='Force full recalculation from scratch, removing any previous '
                   'run artefacts (dist0.npy, dist1.npy, ordering.npy).')
def phierCC(profile, n_proc, clustering_method, allowed_missing, clean):
    """
    pHierCC functions takes a file containing allelic profiles (as in https://pubmlst.org/data/), calculates
    distance between each profile (dual_dist function from getDistance) and performs
    hierarchical clustering of the full dataset based on a minimum-spanning (or macimum) tree.

    When dist0.npy, dist1.npy and ordering.npy all exist in the output directory
    (from a previous run), incremental mode is activated: old distances are
    reused and only pairs involving new STs are computed.
    """

    output_dir = os.path.dirname(profile)
    if not output_dir:
        output_dir = '.'
    profile_file = profile
    numpy_dist0_out = f'{output_dir}/dist0.npy'
    numpy_dist1_out = f'{output_dir}/dist1.npy'
    ordering_path = f'{output_dir}/ordering.npy'

    if clean:
        for f in (numpy_dist0_out, numpy_dist1_out, ordering_path):
            if os.path.exists(f):
                os.remove(f)
                logging.info(f'--clean: removed {f}')

    # Read profiles file
    mat, names = prepare_mat(profile_file)
    n_loci = mat.shape[1] - 1

    # Build a stable lookup from mat row index → original name.
    # For numeric IDs names == mat.T[0]; for text IDs (e.g. "local_1")
    # mat.T[0] holds synthetic sequential integers while names keeps the
    # original strings.  We always use names for tracking between runs.
    matid_to_name = {int(mat[i, 0]): str(names[i]) for i in range(mat.shape[0])}

    # --- Decide: incremental or full mode ---
    incremental = False
    old_n = 0
    if (os.path.exists(ordering_path)
            and os.path.exists(numpy_dist0_out)
            and os.path.exists(numpy_dist1_out)):

        old_ordering = np.load(ordering_path, allow_pickle=True)
        old_n = len(old_ordering)
        old_ordering_str = [str(x) for x in old_ordering]

        new_name_set = set(str(n) for n in names)
        missing_sts = [st for st in old_ordering_str if st not in new_name_set]

        if old_n > mat.shape[0]:
            logging.warning(
                f'New profile has fewer STs ({mat.shape[0]}) than previous '
                f'run ({old_n}). Falling back to full mode.')
        elif old_n == mat.shape[0]:
            if missing_sts:
                logging.warning(
                    f'{len(missing_sts)} old STs replaced by new ones '
                    f'(e.g. {missing_sts[:5]}). Falling back to full mode.')
            else:
                logging.info(
                    'Profile unchanged – same STs as previous run. '
                    'Skipping calculations.')
                sys.exit(42)
        else:
            if missing_sts:
                logging.warning(
                    f'{len(missing_sts)} old STs missing from new profile '
                    f'(e.g. {missing_sts[:5]}). Falling back to full mode.')
            else:
                expected_size = old_n * (old_n - 1) // 2
                probe = np.load(numpy_dist0_out, mmap_mode='r', allow_pickle=True)
                if probe.shape[0] != expected_size:
                    logging.warning(
                        f'Old dist0 size {probe.shape[0]} != expected {expected_size}. '
                        f'Falling back to full mode.')
                    del probe
                else:
                    del probe
                    incremental = True

    if incremental:
        n_new_sts = mat.shape[0] - old_n
        logging.info(
            f'Incremental mode: {old_n} existing STs + {n_new_sts} new STs')

        old_name_to_pos = {st: pos for pos, st in enumerate(old_ordering_str)}
        name_for_row = [matid_to_name[int(row[0])] for row in mat]

        old_mask = np.array([n in old_name_to_pos for n in name_for_row])
        new_mask = ~old_mask

        old_rows = mat[old_mask]
        old_names_subset = [n for n, m in zip(name_for_row, old_mask) if m]
        old_sort = np.array([old_name_to_pos[n] for n in old_names_subset])
        old_rows = old_rows[np.argsort(old_sort)]

        new_rows = mat[new_mask]
        new_names_subset = [n for n, m in zip(name_for_row, new_mask) if m]

        new_public, new_local, names_pub, names_loc = _split_local(
            new_rows, new_names_subset)
        new_rows = np.vstack([new_public, new_local]) if len(new_local) else new_public
        sorted_new_names = names_pub + names_loc

        mat = np.vstack([old_rows, new_rows])

        sorted_old_names = [old_ordering_str[i] for i in np.argsort(old_sort)]
        ordered_names = sorted_old_names + sorted_new_names
    else:
        mat_names = [str(names[i]) for i in range(len(names))]
        pub_rows, loc_rows, pub_names, loc_names = _split_local(
            mat, mat_names)

        if len(loc_rows):
            mat = np.vstack([pub_rows, loc_rows])
        else:
            mat = pub_rows
        ordered_names = pub_names + loc_names

    np.save(ordering_path, np.array(ordered_names, dtype=object),
            allow_pickle=True, fix_imports=True)

    logging.info(
        'Loaded in allelic profiles with dimension: {0} and {1}. '
        'The first column is assumed to be type id.'.format(*mat.shape))

    start = 0

    # ---- Distance matrix 0 (condensed / squareform) ----

    if incremental:
        logging.info('Expanding distance matrix 0 (incremental)')
        dist = ExpandSquareformParallel(numpy_dist0_out, old_n, mat,
                                        n_proc, allowed_missing)
        logging.info(f'Saving distance matrix 0 to {numpy_dist0_out}')
        np.save(numpy_dist0_out, dist, allow_pickle=True, fix_imports=True)
    else:
        logging.info('Calculate distance matrix 0 (full)')
        dist = GetSquareformParallel(mat, n_proc, allowed_missing)
        logging.info(f'Saving distance matrix 0 to {numpy_dist0_out}')
        np.save(numpy_dist0_out, dist, allow_pickle=True, fix_imports=True)

    # ---- Phase 1: hierarchical clustering (uses dist0) ----

    res_template = np.repeat(mat.T[0], int(mat.shape[1]) + 1).reshape(mat.shape[0], -1)
    res_template[res_template < 0] = np.max(mat.T[0]) + 100
    res_template.T[0] = mat.T[0]

    index = {s: i for i, s in enumerate(mat.T[0])}
    results = {}

    for method in clustering_method:
        logging.info(f'Start {method} linkage clustering')
        res = res_template.copy()
        slc = linkage(dist, method=method)

        descendents = [[m] for m in mat.T[0]] + [None for _ in np.arange(mat.shape[0] - 1)]
        for idx, c in enumerate(slc.astype(int)):
            n_id = idx + mat.shape[0]
            d = sorted([int(c[0]), int(c[1])], key=lambda x: descendents[x][0])
            min_id = min(descendents[d[0]])
            descendents[n_id] = descendents[d[0]] + descendents[d[1]]
            for tgt in descendents[d[1]]:
                res[index[tgt], c[2] + 1:] = res[index[min_id], c[2] + 1:]

        results[method] = res

    del dist

    # ---- Phase 2: attach genomes and save (uses dist1) ----

    if incremental:
        logging.info('Expanding distance matrix 1 (incremental)')
        dist = ExpandDistanceParallel(numpy_dist1_out, old_n, mat,
                                       n_proc, allowed_missing, depth=1)
        logging.info(f'Saving distance matrix 1 to {numpy_dist1_out}')
        np.save(numpy_dist1_out, dist, allow_pickle=True, fix_imports=True)
    else:
        logging.info('Calculate distance matrix 1 (full)')
        dist = GetDistanceParallel(mat, n_proc, start, allowed_missing, depth=1)
        logging.info(f'Saving distance matrix 1 to {numpy_dist1_out}')
        np.save(numpy_dist1_out, dist, allow_pickle=True, fix_imports=True)

    for method in clustering_method:
        res = results[method]
        logging.info(f'Attach genomes onto the tree ({method}).')
        for id, (r, d) in enumerate(zip(res[start:], dist[:, :, 0])):
            if id + start > 0:
                i = np.argmin(d[:id+start])
                min_d = d[i]
                if r[min_d + 1] > res[i, min_d + 1]:
                    r[min_d + 1:] = res[i, min_d + 1:]

        res.T[0] = mat.T[0]
        res = res[np.argsort(res.T[0])]

        with gzip.open(f'{output_dir}/profile_{method}_linkage.HierCC.gz', 'wt') as fout:
            fout.write('#ST_id\t{0}\n'.format('\t'.join(['HC' + str(id) for id in np.arange(n_loci + 1)])))
            for n, r in zip(names, res):
                fout.write('\t'.join([str(n)] + [str(rr) for rr in r[1:]]) + '\n')
        prep_index(f'{output_dir}/profile_{method}_linkage.HierCC.gz')
        logging.info(f'Saving clustering results to profile_{method}_linkage.HierCC.gz')

    del dist
    return True

if __name__ == '__main__':
    phierCC(sys.argv[1:])

