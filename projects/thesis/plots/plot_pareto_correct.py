"""Per-system Pareto-front plot with truth-matching markers.

For each system that has at least one structurally-successful rep,
pick the first such rep and plot its Pareto-0 solutions in objective
space. Points with ``hamming == 0`` (structurally correct, possibly
via a ``truth_alternatives`` form) are drawn in red; other Pareto-0
points in grey.

Output: one PNG per system at
``projects/thesis/figures/pareto_<system>.png``, plus a grid montage
at ``projects/thesis/figures/pareto_grid.png``.

For coupled systems (lv, lorenz, ns) the per-equation
``(discrepancy, instability)`` pairs are flattened: each Pareto-0
solution contributes one point per equation. The same solution-level
``hamming == 0`` flag colours all of that solution's per-equation
points so the reader can see the trade-off slice for the matching
solution.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
_FIG_DIR = os.path.join(_PROJECT_DIR, 'figures')


SYSTEM_ORDER = [
    'ode', 'vdp', 'lorenz', 'lv',
    'ac', 'burgers_inviscid', 'burgers_viscous',
    'kdv', 'kdv_cossin', 'ks',
    'wave', 'pde_compound', 'pde_divide',
]


def _find_first_success(system: str):
    """Return the first rep JSON for ``system`` with structural_success=True.

    Falls back to ``None`` if no successful rep exists.
    """
    pattern = os.path.join(_RESULTS_DIR, system, 'new_rep*.json')
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get('structural_success'):
            return rec, path
    return None, None


def _flatten_objectives(objs_per_solution):
    """Convert per-solution objective vector(s) to a flat (n_points, 2) array.

    Each per-solution entry may be:
    - length 2 (single-equation system: [discrepancy, complexity/instability]); or
    - length 2*k (k-equation system: [d1, i1, d2, i2, ..., dk, ik]).

    Returns a list of (per_solution_index, np.ndarray of shape (k, 2))
    so the caller can colour all of a solution's points uniformly.
    """
    out = []
    for sidx, obj in enumerate(objs_per_solution):
        if obj is None:
            continue
        arr = np.asarray(obj, dtype=float).reshape(-1)
        if arr.size == 0 or arr.size % 2 != 0:
            continue
        pairs = arr.reshape(-1, 2)
        out.append((sidx, pairs))
    return out


def plot_one(ax, rec: dict, system: str, rep_path: str):
    """Render one system's Pareto-0 onto ``ax``."""
    objs = rec.get('objectives_per_solution') or []
    hams = rec.get('hamming_per_solution') or []
    flat = _flatten_objectives(objs)
    if not flat:
        ax.text(0.5, 0.5, 'no objectives recorded',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(system)
        return

    # Colour solutions whose hamming == 0 in red; others grey. For coupled
    # systems where the metric matches via truth_alternatives, hamming == 0
    # still flags "this is the truth-matching solution".
    matched_xs, matched_ys = [], []
    other_xs, other_ys = [], []
    for sidx, pairs in flat:
        is_match = (sidx < len(hams) and hams[sidx] == 0)
        target_xs = matched_xs if is_match else other_xs
        target_ys = matched_ys if is_match else other_ys
        for d, c in pairs:
            target_xs.append(float(d))
            target_ys.append(float(c))

    if other_xs:
        ax.scatter(other_xs, other_ys, s=40, c='#bbbbbb',
                   edgecolors='#666', linewidths=0.4,
                   label=f'other ({len(other_xs)})', zorder=2)
    if matched_xs:
        ax.scatter(matched_xs, matched_ys, s=80, c='#d62728',
                   edgecolors='black', linewidths=0.6, marker='*',
                   label=f'truth-match ({len(matched_xs)})', zorder=3)
    ax.set_xscale('symlog', linthresh=1e-12)
    ax.set_yscale('symlog', linthresh=1e-12)
    ax.grid(True, which='major', alpha=0.3)
    n_sol = rec.get('n_pareto_solutions', len(objs))
    seed = rec.get('seed')
    ax.set_title(f'{system} (seed {seed}, {n_sol} Pareto-0 sols)', fontsize=10)
    ax.set_xlabel('discrepancy', fontsize=9)
    ax.set_ylabel('complexity / instability', fontsize=9)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=_FIG_DIR,
                   help='Directory for per-system + grid PNGs.')
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    panels = []
    for system in SYSTEM_ORDER:
        rec, path = _find_first_success(system)
        if rec is None:
            print(f'  [skip] {system}: no structurally-successful rep')
            continue
        # Individual figure.
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        plot_one(ax, rec, system, path)
        out_path = os.path.join(args.out_dir, f'pareto_{system}.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        panels.append((system, rec, path))
        print(f'  wrote {out_path}')

    # Grid montage.
    n = len(panels)
    if n == 0:
        print('  no panels to montage')
        return 0
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows),
                              squeeze=False)
    for i, (system, rec, path) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        plot_one(ax, rec, system, path)
    # Blank the trailing empty axes.
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle('EPDE NEW Pareto-0 per system — '
                 'red star = structurally-correct solution', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = os.path.join(args.out_dir, 'pareto_grid.png')
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f'  wrote {grid_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
