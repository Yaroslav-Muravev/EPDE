"""Re-evaluate existing rep JSONs under the target-side-independent metric.

The :mod:`thesis_metrics` ``canonical_tokens`` / ``hamming`` /
``structural_success`` were updated so that an equation's canonical
form is its unordered set of all terms (target + rhs combined). Under
the old (target-aware) metric, equations whose RPS picked a different
target term than the truth were scored as wrong (high Hamming,
``structural_success=False``) even when algebraically identical.

This script walks every ``*_rep*.json`` under
``projects/thesis/results/`` and, for each file:

1. Reads the original ``discovered_text_per_solution`` and the system's
   ``truth_equations`` from ``configs/<system>.yaml``.
2. Recomputes ``hamming_per_solution`` and ``structural_success`` with
   the new metric.
3. Updates the file in place ONLY when the new ``structural_success``
   flips from ``False`` to ``True`` (the case the user asked for: a
   correct equation previously tagged wrong because the target term
   differed). All other fields are left untouched.

Print a one-line summary per affected file plus a final tally.

Usage:
    python projects/thesis/rescore_results.py [--results-dir results] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import yaml  # type: ignore  # noqa: E402

from thesis_metrics import (  # noqa: E402
    canonical_tokens, hamming_best,
)


def _load_truth_alternatives(system: str, configs_dir: str):
    """Return (primary_eqs, [alt_eqs, ...]) from the per-system YAML.

    Primary = ``truth_equations`` list; alternatives = each entry of
    ``truth_alternatives`` (list-of-lists). Returns ``None`` if the
    primary is absent (system has no truth declared).
    """
    path = os.path.join(configs_dir, f'{system}.yaml')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    primary = list(cfg.get('truth_equations') or [])
    if not primary:
        return None
    alts = [list(eqs) for eqs in (cfg.get('truth_alternatives') or []) if eqs]
    return primary, alts


def _extract_text_per_solution(rec: dict) -> list[list[str]]:
    """Return per-solution list-of-strings discovered text.

    Older / current JSONs store this under ``discovered_text_per_solution``
    as either a list of lists (one list[str] per solution) OR (legacy) a
    flat list of strings for a single solution. Normalise both.
    """
    raw = rec.get('discovered_text_per_solution') or []
    if not raw:
        return []
    if isinstance(raw[0], str):
        return [list(raw)]
    return [list(item) if isinstance(item, list) else [str(item)]
            for item in raw]


def rescore_file(path: str, configs_dir: str, dry_run: bool) -> tuple[bool, str]:
    """Return ``(changed, summary_line)``."""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            rec = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f'  SKIP  {path}: unreadable ({exc!r})'

    system = rec.get('system')
    if not system:
        return False, f'  SKIP  {path}: no "system" field'

    truth_info = _load_truth_alternatives(system, configs_dir)
    if truth_info is None:
        return False, f'  SKIP  {path}: no truth_equations for {system}'
    primary_eqs, alt_eq_lists = truth_info

    sol_texts = _extract_text_per_solution(rec)
    if not sol_texts:
        return False, f'  SKIP  {path}: no discovered_text_per_solution'

    canon_alternatives = (canonical_tokens(primary_eqs),) + tuple(
        canonical_tokens(eqs) for eqs in alt_eq_lists
    )
    new_hammings = []
    for sol in sol_texts:
        canon_disc = canonical_tokens(sol)
        new_hammings.append(hamming_best(canon_disc, canon_alternatives))

    if not new_hammings:
        return False, f'  SKIP  {path}: empty new_hammings'

    new_best_idx = int(min(range(len(new_hammings)), key=lambda i: new_hammings[i]))
    new_best_h = new_hammings[new_best_idx]
    new_struct = (new_best_h == 0)

    old_struct = bool(rec.get('structural_success', False))
    if old_struct or not new_struct:
        # Only touch files that flipped False -> True.
        return False, ''

    old_h = rec.get('hamming')
    rec['hamming_per_solution'] = new_hammings
    rec['hamming'] = new_best_h
    rec['structural_success'] = True
    # Reflect the new best solution in the scalar fields if it differs.
    if 'discovered_text_per_solution' in rec and len(rec['discovered_text_per_solution']) > new_best_idx:
        rec['discovered_text'] = rec['discovered_text_per_solution'][new_best_idx]
    if 'discovered_tokens_per_solution' in rec and len(rec['discovered_tokens_per_solution']) > new_best_idx:
        rec['discovered_tokens'] = rec['discovered_tokens_per_solution'][new_best_idx]
    if 'discovery_epoch_per_solution' in rec and len(rec['discovery_epoch_per_solution']) > new_best_idx:
        rec['discovery_epoch'] = rec['discovery_epoch_per_solution'][new_best_idx]
    if 'objectives_per_solution' in rec and len(rec['objectives_per_solution']) > new_best_idx:
        rec['objectives'] = rec['objectives_per_solution'][new_best_idx]

    summary = (f'  FLIP  {path}: '
               f'hamming {old_h} -> {new_best_h}, '
               f'structural_success False -> True')
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=2, default=str)
    return True, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir',
                   default=os.path.join(_THIS_DIR, 'results'),
                   help='Root directory to walk for rep JSONs.')
    p.add_argument('--configs-dir',
                   default=os.path.join(_THIS_DIR, 'configs'),
                   help='YAML configs root.')
    p.add_argument('--dry-run', action='store_true',
                   help='Report flips but do not modify files.')
    args = p.parse_args(argv)

    pattern = os.path.join(args.results_dir, '**', '*_rep*.json')
    paths = [p for p in glob.glob(pattern, recursive=True)
             if not p.endswith('.history.json')]

    print(f'Scanning {len(paths)} rep JSONs under {args.results_dir} '
          f'({"DRY RUN" if args.dry_run else "writing changes"})')
    flipped = 0
    for path in sorted(paths):
        changed, summary = rescore_file(path, args.configs_dir, args.dry_run)
        if changed:
            flipped += 1
            print(summary)

    print()
    print(f'Total flipped (False -> True): {flipped} / {len(paths)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
