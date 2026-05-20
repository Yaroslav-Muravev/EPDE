"""
Aggregator for thesis Section 4.5 smoke / full-run results.

Walks every ``projects/thesis/results/<system>/<pipeline>_rep<NN>.json``
file, groups by (system, pipeline), and writes a markdown summary plus a
JSON snapshot. Metrics per (system, pipeline) cell:

    - structural_success_rate (with Wilson 95% CI)
    - mean Hamming distance
    - consistency_rate (modal-set agreement)
    - mean runtime

Pass ``--root`` to point at a tagged results tree (e.g.
``projects/thesis/results/ablation_v2``) -- the layout is always
``<root>/<system>/*.json``.
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


PIPELINES = ('legacy', 'new')
DEFAULT_RESULTS_DIR = os.path.join(_THIS_DIR, 'results')


def _load_records(root: str):
    records = defaultdict(lambda: defaultdict(list))  # records[system][pipeline] -> list
    pattern = os.path.join(root, '*', '*.json')
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        system = rec.get('system') or os.path.basename(os.path.dirname(path))
        pipeline = rec.get('pipeline')
        if pipeline not in PIPELINES:
            continue
        records[system][pipeline].append(rec)
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


def _format_table(summary: dict) -> str:
    header = (
        '| System | n | Legacy success | Legacy H | Legacy cons | '
        'NEW success | NEW H | NEW cons | runtime (L / N) |'
    )
    sep = '|---|---|---|---|---|---|---|---|---|'
    rows = [header, sep]
    for system in sorted(summary.keys()):
        legacy = summary[system].get('legacy', {'n': 0})
        new = summary[system].get('new', {'n': 0})

        def cell_success(c):
            if c['n'] == 0:
                return '-'
            return (
                f"{c['rate']*100:.0f}% [{c['wilson_lo']*100:.0f}-{c['wilson_hi']*100:.0f}%] "
                f"({c['successes']}/{c['n']})"
            )

        def cell_num(c, key, fmt):
            if c['n'] == 0:
                return '-'
            v = c.get(key)
            if v is None or (isinstance(v, float) and v != v):
                return '-'
            return fmt.format(v)

        rows.append(
            f"| {system} | {max(legacy['n'], new['n'])} | "
            f"{cell_success(legacy)} | {cell_num(legacy, 'mean_hamming', '{:.1f}')} | "
            f"{cell_num(legacy, 'consistency', '{:.2f}')} | "
            f"{cell_success(new)} | {cell_num(new, 'mean_hamming', '{:.1f}')} | "
            f"{cell_num(new, 'consistency', '{:.2f}')} | "
            f"{cell_num(legacy, 'mean_runtime_sec', '{:.1f}')}s / "
            f"{cell_num(new, 'mean_runtime_sec', '{:.1f}')}s |"
        )
    return '\n'.join(rows)


def aggregate(root: str = None) -> dict:
    root = root or DEFAULT_RESULTS_DIR
    records = _load_records(root)
    summary = {
        system: {pipeline: _summarize_cell(reps) for pipeline, reps in by_pipeline.items()}
        for system, by_pipeline in records.items()
    }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_RESULTS_DIR,
                        help=f"results root to scan (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument('--out', default=None,
                        help="path for the JSON snapshot (default: <root>/../thesis_summary.json)")
    args = parser.parse_args(argv)

    summary = aggregate(args.root)
    print(_format_table(summary))
    out_path = args.out or os.path.join(_THIS_DIR, 'thesis_summary.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
