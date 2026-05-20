"""
Aggregator for the thesis Section 4.5 ablation study (2x2x2 factorial).

Walks every ``projects/thesis/results/<system>/<cell>_rep<NN>.json`` file
whose ``pipeline`` field names one of the 8 ablation cells, groups by
(system, cell), and writes a markdown summary plus a JSON snapshot.

If your ablation runs landed under a tag (``--outdir ablation_v2`` ->
``results/ablation_v2/<system>/``), point ``--root`` at that subtree.
Either way the layout is always ``<root>/<system>/*.json``; the 000
(``legacy``) and 111 (``new``) corners are read from the same JSON files
that ``thesis_aggregate.py`` already consumes.

Cell-label semantics (each label lists the NEW components that are ON):

    legacy        000  fitness=L2,   sparsity=LASSO, use_pic=False
    wape          100  fitness=L2LR, sparsity=LASSO, use_pic=False
    instab        010  fitness=L2,   sparsity=LASSO, use_pic=True
    reg           001  fitness=L2,   sparsity=VWSR,  use_pic=False
    wape_instab   110  fitness=L2LR, sparsity=LASSO, use_pic=True
    wape_reg      101  fitness=L2LR, sparsity=VWSR,  use_pic=False
    instab_reg    011  fitness=L2,   sparsity=VWSR,  use_pic=True
    new           111  fitness=L2LR, sparsity=VWSR,  use_pic=True
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from thesis_metrics import consistency_rate, wilson_ci  # noqa: E402


# Ordered so the report reads from "all off" to "all on" along each axis.
ABLATION_CELLS = (
    'legacy',
    'wape', 'instab', 'reg',
    'wape_instab', 'wape_reg', 'instab_reg',
    'new',
)

DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, 'results')


def _load_records(root: str):
    records = defaultdict(lambda: defaultdict(list))  # records[system][cell] -> list
    pattern = os.path.join(root, '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        system = rec.get('system') or os.path.basename(os.path.dirname(path))
        cell = rec.get('pipeline')
        if cell not in ABLATION_CELLS:
            continue
        records[system][cell].append(rec)
    return records


def _summarize_cell(reps: list) -> dict:
    if not reps:
        return {'n': 0}
    successes = sum(1 for r in reps if r.get('structural_success'))
    hammings = [r['hamming'] for r in reps if r.get('hamming') is not None]
    runtimes = [r['runtime_sec'] for r in reps if 'runtime_sec' in r]
    discovered_tokens = [json.dumps(r.get('discovered_tokens', []), sort_keys=True) for r in reps]
    rate = successes / len(reps)
    ci = wilson_ci(successes, len(reps))
    mean_h = statistics.fmean(hammings) if hammings else float('nan')
    mean_t = statistics.fmean(runtimes) if runtimes else float('nan')
    errors = sum(1 for r in reps if 'error' in r)
    return {
        'n': len(reps),
        'successes': successes,
        'rate': rate,
        'wilson_lo': ci[0],
        'wilson_hi': ci[1],
        'mean_hamming': mean_h,
        'consistency': consistency_rate(discovered_tokens),
        'mean_runtime_sec': mean_t,
        'errors': errors,
    }


def _cell_axes(cell: str) -> tuple:
    """Return ``(wape_on, instab_on, reg_on)`` triple for a given cell label."""
    if cell == 'legacy':
        return (False, False, False)
    if cell == 'new':
        return (True, True, True)
    parts = set(cell.split('_'))
    return ('wape' in parts, 'instab' in parts, 'reg' in parts)


def _format_table(summary: dict) -> str:
    header = (
        '| System | Cell | W | I | R | n | Success | mean H | cons | runtime |'
    )
    sep = '|---|---|---|---|---|---|---|---|---|---|'
    rows = [header, sep]

    def _check(b: bool) -> str:
        return 'X' if b else '.'

    def _success(c):
        if c['n'] == 0:
            return '-'
        return (
            f"{c['rate']*100:.0f}% [{c['wilson_lo']*100:.0f}-{c['wilson_hi']*100:.0f}%] "
            f"({c['successes']}/{c['n']})"
        )

    def _num(c, key, fmt):
        if c['n'] == 0:
            return '-'
        v = c.get(key)
        if v is None or (isinstance(v, float) and v != v):
            return '-'
        return fmt.format(v)

    for system in sorted(summary.keys()):
        for cell in ABLATION_CELLS:
            c = summary[system].get(cell, {'n': 0})
            w, i, r = _cell_axes(cell)
            rows.append(
                f"| {system} | {cell} | {_check(w)} | {_check(i)} | {_check(r)} | "
                f"{c['n']} | {_success(c)} | "
                f"{_num(c, 'mean_hamming', '{:.1f}')} | "
                f"{_num(c, 'consistency', '{:.2f}')} | "
                f"{_num(c, 'mean_runtime_sec', '{:.1f}')}s |"
            )
    return '\n'.join(rows)


def _format_contributions(summary: dict) -> str:
    """Render the marginal contribution of each axis per system."""
    axes = (
        ('WAPE',   0, [('legacy', 'wape'), ('instab', 'wape_instab'),
                        ('reg', 'wape_reg'), ('instab_reg', 'new')]),
        ('Instab', 1, [('legacy', 'instab'), ('wape', 'wape_instab'),
                        ('reg', 'instab_reg'), ('wape_reg', 'new')]),
        ('Reg',    2, [('legacy', 'reg'), ('wape', 'wape_reg'),
                        ('instab', 'instab_reg'), ('wape_instab', 'new')]),
    )
    rows = ['| System | Axis | mean delta success | mean delta H | n pairs |',
            '|---|---|---|---|---|']
    for system in sorted(summary.keys()):
        for axis_name, _idx, pairs in axes:
            d_rate = []
            d_h = []
            for off_cell, on_cell in pairs:
                off = summary[system].get(off_cell, {'n': 0})
                on = summary[system].get(on_cell, {'n': 0})
                if off['n'] == 0 or on['n'] == 0:
                    continue
                d_rate.append(on['rate'] - off['rate'])
                if (
                    on.get('mean_hamming') is not None
                    and off.get('mean_hamming') is not None
                    and on['mean_hamming'] == on['mean_hamming']
                    and off['mean_hamming'] == off['mean_hamming']
                ):
                    d_h.append(on['mean_hamming'] - off['mean_hamming'])
            if not d_rate:
                rows.append(f"| {system} | {axis_name} | - | - | 0 |")
                continue
            mean_dr = statistics.fmean(d_rate)
            mean_dh = statistics.fmean(d_h) if d_h else float('nan')
            dh_str = f"{mean_dh:+.2f}" if mean_dh == mean_dh else '-'
            rows.append(
                f"| {system} | {axis_name} | {mean_dr*100:+.1f}pp | "
                f"{dh_str} | {len(d_rate)} |"
            )
    return '\n'.join(rows)


def aggregate(root: str = None) -> dict:
    root = root or DEFAULT_RESULTS_DIR
    records = _load_records(root)
    summary = {
        system: {cell: _summarize_cell(reps) for cell, reps in by_cell.items()}
        for system, by_cell in records.items()
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_RESULTS_DIR,
                        help=f"results root to scan (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument('--out', default=None,
                        help="path for the JSON snapshot (default: <thesis>/thesis_ablation_summary.json)")
    args = parser.parse_args(argv)

    summary = aggregate(args.root)
    print('# Thesis Section 4.5 -- Ablation Cells')
    print()
    print(_format_table(summary))
    print()
    print('# Marginal contribution per axis (mean delta across the 4 mutually-exclusive pairs)')
    print()
    print(_format_contributions(summary))
    out_path = args.out or os.path.join(_THIS_DIR, 'thesis_ablation_summary.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
