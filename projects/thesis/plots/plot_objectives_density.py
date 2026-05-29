"""Plot per-equation objective-space density of all candidates throughout the search.

For each system, loads every rep JSON under ``results/rerun/<system>/``,
reads the sidecar ``<rep>.history.json`` (full ``MOEADDOptimizer._hist``
snapshots — all candidates across all Pareto levels, per epoch), and
produces a ``1 x n_equations`` figure where each panel is a 2D KDE in
log-log objective space (discrepancy_i, complexity_i) overlaid for the
configured pipelines (``legacy`` vs ``new`` by default).

Usage:
    python projects/thesis/plots/plot_objectives_density.py [system ...]
           [--results-dir projects/thesis/results/rerun]
           [--pipelines legacy new]
           [--out-dir projects/thesis/figures]
           [--max-points 50000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from collections import defaultdict
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THESIS_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_REPO_ROOT = os.path.abspath(os.path.join(_THESIS_DIR, '..', '..'))
DEFAULT_RESULTS_DIR = os.path.join(_THESIS_DIR, 'results')
DEFAULT_OUT_DIR = os.path.join(_THESIS_DIR, 'figures')

# Pipeline -> color mapping. Matches the conventional "legacy = warm,
# new = cool" pairing used throughout the thesis figures.
PIPELINE_COLORS = {
    'legacy': '#e07b39',
    'new': '#3a7cb8',
    'wape': '#9c6ade',
    'instab': '#5fb46a',
    'reg': '#c43d6a',
    'wape_instab': '#d4a017',
    'wape_reg': '#6c9aa0',
    'instab_reg': '#7a4b6e',
}

# Pipeline -> second-objective name. The 2nd MOEA/D objective depends
# on ``use_pic``: True -> equation_terms_stability ("instability"),
# False -> equation_complexity_by_factors ("complexity"). See the
# 2x2x2 factorial table in projects/thesis/thesis_runner.py.
PIPELINE_Y_LABEL = {
    'legacy':       'complexity',
    'wape':         'complexity',
    'instab':       'instability',
    'reg':          'complexity',
    'wape_instab':  'instability',
    'wape_reg':     'complexity',
    'instab_reg':   'instability',
    'new':          'instability',
}


def _load_rep(rep_path: str) -> dict | None:
    """Load a rep JSON; return ``None`` on parse failure."""
    try:
        with open(rep_path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"could not load {rep_path}: {exc!r}")
        return None


def _load_history(rep_dir: str, history_basename: str) -> list | None:
    """Load a sidecar history file; return ``None`` if missing/invalid."""
    history_path = os.path.join(rep_dir, history_basename)
    try:
        with open(history_path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        return payload.get('candidate_history')
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"could not load history {history_path}: {exc!r}")
        return None


def gather_points(
    system: str, pipeline: str, results_dir: str,
) -> tuple[np.ndarray, int, int]:
    """Concatenate all-candidate objective vectors from all reps' sidecars.

    Returns ``(points, n_reps, n_reps_with_history)``. ``points`` has
    shape ``(M, n_objectives)`` where M is the total number of
    candidate snapshots across all epochs of all reps that had a
    sidecar. Reps without a sidecar are counted in ``n_reps`` but not
    in ``n_reps_with_history`` (and contribute zero points).
    """
    pattern = os.path.join(results_dir, system, f"{pipeline}_rep*.json")
    rep_paths = sorted(glob.glob(pattern))
    # Exclude the sidecar files themselves (they also match *_rep*.json
    # because the basename is e.g. legacy_rep00.history.json).
    rep_paths = [p for p in rep_paths if not p.endswith('.history.json')]

    chunks: list[np.ndarray] = []
    n_with_hist = 0
    for rep_path in rep_paths:
        rec = _load_rep(rep_path)
        if rec is None:
            continue
        hist_basename = rec.get('history_path')
        if not hist_basename:
            continue
        rep_dir = os.path.dirname(rep_path)
        candidate_history = _load_history(rep_dir, hist_basename)
        if not candidate_history:
            continue
        for snapshot in candidate_history:
            if not snapshot:
                continue
            arr = np.asarray(snapshot, dtype=float)
            if arr.ndim != 2 or arr.size == 0:
                continue
            chunks.append(arr)
        n_with_hist += 1

    if not chunks:
        return np.empty((0, 0)), len(rep_paths), 0
    points = np.vstack(chunks)
    return points, len(rep_paths), n_with_hist


def _safe_log10(values: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(values, floor))


def _scatter_logspace(
    ax, x_lin: np.ndarray, y_lin: np.ndarray,
    color: str, label: str,
) -> int:
    """Draw a per-pipeline scatter in log-log space.

    Filters non-finite points and renders the surviving candidates as a
    semi-transparent scatter. Density is read from point overplotting;
    no KDE / contour is drawn. Returns the rendered point count so the
    caller can surface it in the panel title.
    """
    lx = _safe_log10(x_lin)
    ly = _safe_log10(y_lin)

    mask = np.isfinite(lx) & np.isfinite(ly)
    lx = lx[mask]
    ly = ly[mask]
    if lx.size == 0:
        return 0
    ax.scatter(10 ** lx, 10 ** ly, color=color, alpha=0.45, s=14,
               edgecolors='none', label=label)
    return int(lx.size)


def _downsample(points: np.ndarray, cap: int) -> np.ndarray:
    """Random downsample to ``cap`` rows if needed."""
    if cap <= 0 or points.shape[0] <= cap:
        return points
    idx = np.random.default_rng(0).choice(points.shape[0], cap, replace=False)
    return points[idx]


def _plot_one_pipeline(
    system: str, pipeline: str, points: np.ndarray,
    n_reps: int, n_with_hist: int, out_dir: str,
) -> str:
    """Render one figure for one (system, pipeline) and return the path.

    Each pipeline gets its own figure because legacy and new use
    different objective definitions (L2Fitness vs L2LRFitness;
    LASSO vs VWSR complexity) -- overlaying them on shared axes is
    misleading.
    """
    n_obj = points.shape[1]
    if n_obj % 2 != 0:
        warnings.warn(f"[{system}/{pipeline}] n_objectives={n_obj} "
                      f"not divisible by 2; plot pairs may be miscoded")
    n_equations = max(1, n_obj // 2)

    # Single pipeline per figure; share x/y so the per-equation panels
    # of coupled systems (LV / Lorenz / NS) compare on identical
    # log-log axes.
    fig, axes = plt.subplots(1, n_equations,
                             figsize=(5.2 * n_equations, 4.6),
                             squeeze=False,
                             sharex=True, sharey=True)
    axes = axes[0]

    color = PIPELINE_COLORS.get(pipeline, '#444444')
    label = f"{pipeline}  ({n_with_hist}/{n_reps} reps)"
    y_axis_name = PIPELINE_Y_LABEL.get(pipeline, 'complexity')

    first_drawn_ax = None
    for eq_idx, ax in enumerate(axes):
        ix = 2 * eq_idx
        iy = 2 * eq_idx + 1
        if iy >= n_obj:
            ax.set_visible(False)
            continue
        n_pts = _scatter_logspace(ax, points[:, ix], points[:, iy],
                                  color=color, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"discrepancy (eq {eq_idx})")
        ax.set_ylabel(f"{y_axis_name} (eq {eq_idx})")
        ax.grid(alpha=0.3, which='both', linewidth=0.4)
        ax.set_title(f"Equation {eq_idx}  (n={n_pts})")
        if n_pts and first_drawn_ax is None:
            first_drawn_ax = ax

    fig.suptitle(f"{system.upper()} / {pipeline} — candidates throughout "
                 f"search ({points.shape[0]} samples)")
    fig.tight_layout()
    # Single figure-level legend so coupled-system figures (LV / Lorenz /
    # NS) don't repeat the pipeline label per equation panel. Anchored
    # above the figure so it doesn't compete with the data plane.
    if first_drawn_ax is not None:
        handles, labels = first_drawn_ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center',
                       bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                       frameon=False, fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"objectives_density_{system}_{pipeline}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  [{system}/{pipeline}] saved {out_path}")
    plt.close(fig)
    return out_path


def plot_system(
    system: str, results_dir: str, pipelines: Iterable[str],
    out_dir: str, max_points: int,
) -> bool:
    """Build one figure per (system, pipeline). Returns True on any success.

    Pipelines are NOT overlaid because legacy and new use different
    objective definitions (different fitness class and different
    sparsity-driven complexity scale). Overlaying them on shared log
    axes would be visually misleading -- separate figures keep each
    pipeline's intrinsic scale honest.
    """
    any_ok = False
    for pipeline in pipelines:
        points, n_reps, n_with_hist = gather_points(system, pipeline, results_dir)
        if points.size == 0:
            print(f"  [{system}/{pipeline}] no usable history "
                  f"({n_reps} rep JSONs found, {n_with_hist} with sidecars)")
            continue
        if max_points and points.shape[0] > max_points:
            print(f"  [{system}/{pipeline}] downsampling "
                  f"{points.shape[0]} -> {max_points}")
            points = _downsample(points, max_points)
        _plot_one_pipeline(system, pipeline, points, n_reps, n_with_hist,
                           out_dir)
        any_ok = True
    if not any_ok:
        print(f"  [{system}] nothing to plot, skipping")
    return any_ok


def _discover_systems(results_dir: str) -> list[str]:
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        name for name in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, name))
        and any(
            fname.endswith('.history.json')
            for fname in os.listdir(os.path.join(results_dir, name))
        )
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('systems', nargs='*',
                   help="Systems to plot (default: auto-discover any subdir "
                        "of --results-dir containing at least one sidecar).")
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--pipelines', nargs='+', default=['legacy', 'new'])
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--max-points', type=int, default=50000)
    args = p.parse_args(argv)

    systems = args.systems or _discover_systems(args.results_dir)
    if not systems:
        print(f"No systems with sidecar histories found under "
              f"{args.results_dir}.")
        return 1

    print(f"Plotting systems: {systems}")
    print(f"Pipelines:        {args.pipelines}")
    print(f"Out dir:          {args.out_dir}")
    any_ok = False
    for system in systems:
        print(f"\n>>> {system}")
        ok = plot_system(system, args.results_dir, args.pipelines,
                         args.out_dir, args.max_points)
        any_ok = any_ok or ok
    return 0 if any_ok else 1


if __name__ == '__main__':
    sys.exit(main())
