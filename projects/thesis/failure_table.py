"""
Per-seed failure table for the thesis Section 4.5 cohort.

For each (system, pipeline) cell, prints a 30-char status string where
position N = outcome of seed N:

    .   structural success (truth matched)
    F   completed but structural mismatch
    C   crashed (no rep file written, or file has ``error`` field)

Also prints a per-cell list of crashed and failed seed numbers so the
user can copy them directly into a re-run command.

Scans the same layout as ``thesis_aggregate.py``:
``<root>/<system>/<pipeline>_rep<NN>.json``. Pipelines covered: the 8
ablation cells (legacy, wape, instab, reg, wape_instab, wape_reg,
instab_reg, new) -- cells with zero reps are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


CELLS = (
    'legacy',
    'wape', 'instab', 'reg',
    'wape_instab', 'wape_reg', 'instab_reg',
    'new',
)
DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, 'results')
N_SEEDS = 30

_REP_RE = re.compile(r'^(?P<cell>[a-z_]+)_rep(?P<seed>\d{2})\.json$')


def _scan(root: str):
    # status[system][cell][seed] in {'S', 'F', 'C'}; default 'C' (missing)
    status = defaultdict(lambda: defaultdict(lambda: {s: 'C' for s in range(N_SEEDS)}))
    pattern = os.path.join(root, '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.history.json'):
            continue
        fname = os.path.basename(path)
        m = _REP_RE.match(fname)
        if not m:
            continue
        cell = m.group('cell')
        if cell not in CELLS:
            continue
        seed = int(m.group('seed'))
        system = os.path.basename(os.path.dirname(path))
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            status[system][cell][seed] = 'C'
            continue
        if 'error' in rec:
            status[system][cell][seed] = 'C'
        elif rec.get('structural_success'):
            status[system][cell][seed] = 'S'
        else:
            status[system][cell][seed] = 'F'
    return status


def _seed_list(status_map: dict, mark: str) -> str:
    seeds = [str(s) for s in sorted(status_map.keys()) if status_map[s] == mark]
    return ','.join(seeds) if seeds else '-'


def _format(status: dict) -> str:
    lines = [
        '# Per-seed failure table',
        '',
        'Legend: `.` = structural success, `F` = completed but wrong, `C` = crashed/missing.',
        'Columns 0..29 of the status string are seeds 0..29.',
        '',
        '| System | Pipeline | Status (seed 0..29) | nS | nF | nC | Crashed seeds | Failed seeds |',
        '|---|---|---|---|---|---|---|---|',
    ]
    systems = sorted(status.keys())
    for system in systems:
        for cell in CELLS:
            if cell not in status[system]:
                continue
            seed_map = status[system][cell]
            string = ''.join('.' if seed_map[s] == 'S' else seed_map[s] for s in range(N_SEEDS))
            n_s = sum(1 for v in seed_map.values() if v == 'S')
            n_f = sum(1 for v in seed_map.values() if v == 'F')
            n_c = sum(1 for v in seed_map.values() if v == 'C')
            crashed = _seed_list(seed_map, 'C')
            failed = _seed_list(seed_map, 'F')
            lines.append(
                f'| {system} | {cell} | `{string}` | {n_s} | {n_f} | {n_c} | '
                f'{crashed} | {failed} |'
            )
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out', default=os.path.join(_THIS_DIR, 'failures.md'))
    args = parser.parse_args(argv)

    status = _scan(args.root)
    text = _format(status)
    print(text)
    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    print(f'\nWrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
