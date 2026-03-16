#!/usr/bin/env python3
"""
Compare two HierCC clustering result files (.HierCC.gz) and report
which STs differ, at which HC thresholds, and whether the differences
are just a shift in the threshold boundary (e.g. cluster merges one
level earlier/later).

Usage:
    python3 compare_hiercc.py <file_a.HierCC.gz> <file_b.HierCC.gz>
"""
import sys, gzip
import pandas as pd
import numpy as np


def load_hiercc(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        df = pd.read_csv(f, sep='\t', dtype=str)
    st_col = df.columns[0]
    df = df.set_index(st_col).sort_index()
    df = df.apply(pd.to_numeric, errors='coerce').astype('Int64')
    return df


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <file_a.HierCC.gz> <file_b.HierCC.gz>')
        sys.exit(1)

    a = load_hiercc(sys.argv[1])
    b = load_hiercc(sys.argv[2])

    common = a.index.intersection(b.index)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)

    print(f'STs in file A: {len(a)}')
    print(f'STs in file B: {len(b)}')
    print(f'STs in common: {len(common)}')
    if len(only_a):
        print(f'STs only in A: {len(only_a)}  (e.g. {list(only_a[:5])})')
    if len(only_b):
        print(f'STs only in B: {len(only_b)}  (e.g. {list(only_b[:5])})')

    a_common = a.loc[common]
    b_common = b.loc[common]
    diff_mask = a_common != b_common

    n_cols = diff_mask.shape[1]
    n_differing_sts = diff_mask.any(axis=1).sum()
    print(f'\nSTs with any difference: {n_differing_sts} / {len(common)}'
          f'  ({100 * n_differing_sts / len(common):.3f}%)')

    cols_affected = diff_mask.sum(axis=0)
    cols_affected = cols_affected[cols_affected > 0]
    if len(cols_affected) == 0:
        print('Files are identical for all common STs.')
        return

    print(f'\nAffected HC columns ({len(cols_affected)} / {n_cols}):')
    print(f'  {"Column":<10} {"STs differ":>12}   {"% of common":>11}')
    print(f'  {"------":<10} {"----------":>12}   {"-----------":>11}')
    for col, cnt in cols_affected.items():
        print(f'  {col:<10} {cnt:>12}   {100*cnt/len(common):>10.3f}%')

    differing_sts = diff_mask.any(axis=1)
    differing_ids = differing_sts[differing_sts].index.tolist()

    print(f'\nPer-ST breakdown (first 20):')
    print(f'  {"ST":<12} {"# cols diff":>11}  {"first diff col":>14}  '
          f'{"last diff col":>14}  detail')
    print(f'  {"--":<12} {"-----------":>11}  {"--------------":>14}  '
          f'{"--------------":>14}  ------')

    for st in differing_ids[:20]:
        row_diff = diff_mask.loc[st]
        changed_cols = row_diff[row_diff].index.tolist()
        n_changed = len(changed_cols)
        first_col = changed_cols[0]
        last_col = changed_cols[-1]

        va = a_common.loc[st, first_col]
        vb = b_common.loc[st, first_col]
        detail = f'{first_col}: {va}→{vb}'

        if n_changed > 1:
            va2 = a_common.loc[st, last_col]
            vb2 = b_common.loc[st, last_col]
            detail += f'  ...  {last_col}: {va2}→{vb2}'

        print(f'  {str(st):<12} {n_changed:>11}  {first_col:>14}  '
              f'{last_col:>14}  {detail}')

    # Detect "threshold shift" pattern: differences are consecutive columns
    # where one file assigns to a lower cluster ID one level earlier
    print(f'\n--- Shift analysis ---')
    shift_count = 0
    non_shift_count = 0
    for st in differing_ids:
        row_a = a_common.loc[st].values
        row_b = b_common.loc[st].values
        diffs = np.where(row_a != row_b)[0]
        if len(diffs) == 0:
            continue
        is_contiguous = (diffs[-1] - diffs[0] + 1) == len(diffs)
        if is_contiguous and diffs[-1] == n_cols - 1:
            shift_count += 1
        else:
            non_shift_count += 1

    print(f'  Threshold-boundary shifts (contiguous diffs reaching last col): '
          f'{shift_count}')
    print(f'  Other patterns: {non_shift_count}')
    if shift_count + non_shift_count > 0:
        print(f'  → {100*shift_count/(shift_count+non_shift_count):.1f}% of '
              f'differing STs are simple boundary shifts')


if __name__ == '__main__':
    main()
