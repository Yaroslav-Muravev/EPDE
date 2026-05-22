"""Profile a single NEW-pipeline run to identify bottlenecks.

Wraps ``thesis_runner.build_search`` in cProfile and post-processes the
stats into three views:

1. Wall-clock phase timing (build_search totals).
2. Top-30 functions by cumulative time and by self-time (tottime).
3. Per-operator aggregation: every ``CompoundOperator.apply`` method,
   summed by owning class.

Usage:
    python projects/thesis/profile_run.py [system ...] [--pipeline new|legacy]
                                          [--seed N] [--epochs N]

Defaults to systems = ['lv', 'wave'], pipeline = 'new', seed = 0,
training_epochs left to the YAML default. ``--epochs`` overrides
``moeadd.training_epochs`` for the run so wall-clock comparisons aren't
distorted by truth-match early-stop variability.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from collections import defaultdict


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from thesis_runner import (  # noqa: E402
    _set_seeds,
    build_search,
    load_config,
    pipeline_settings,
)


def _aggregate_apply_methods(stats: pstats.Stats) -> list:
    """Sum cumulative + total time per ``apply`` method, keyed by ``file:lineno``.

    cProfile records each function as (file, lineno, funcname). For
    ``CompoundOperator.apply`` calls dispatched on subclasses, the
    filename+lineno pins the actual subclass definition. Each (file,
    lineno) pair is a distinct row so subclasses sharing a module
    (e.g., the four mutation operators in mutations.py) don't merge.

    Call this on a *non*-strip_dirs Stats so we can filter by full path
    (only the epde package); the rows themselves are printed with
    truncated paths.
    """
    per_class = defaultdict(lambda: {'cumtime': 0.0, 'tottime': 0.0, 'ncalls': 0, 'key': ''})
    for func_key, (cc, nc, tt, ct, _callers) in stats.stats.items():
        fname, lineno, funcname = func_key
        if funcname != 'apply':
            continue
        fname_norm = fname.replace('\\', '/').lower()
        if '/epde/' not in fname_norm and not fname_norm.endswith('/thesis_runner.py'):
            continue
        short = fname.replace('\\', '/').split('/epde/')[-1]
        key = f"{short}:{lineno}"
        bucket = per_class[key]
        bucket['cumtime'] += ct
        bucket['tottime'] += tt
        bucket['ncalls'] += nc
        bucket['key'] = key
    rows = list(per_class.values())
    rows.sort(key=lambda r: r['cumtime'], reverse=True)
    return rows


def _format_top_n(stats: pstats.Stats, sort_key: str, n: int) -> str:
    buf = io.StringIO()
    stats.stream = buf
    stats.sort_stats(sort_key).print_stats(n)
    return buf.getvalue()


def profile_system(system_name: str, pipeline: str = 'new', seed: int = 0,
                   epochs: int | None = None) -> None:
    cfg = load_config(system_name)
    # Make wall-clock numbers reproducible: turn off the truth-match
    # early stop and (optionally) pin training_epochs.
    cfg.hparams['moeadd']['early_stop_on_truth'] = False
    if epochs is not None:
        cfg.hparams['moeadd']['training_epochs'] = int(epochs)

    pipeline_kwargs = pipeline_settings(pipeline)
    _set_seeds(seed)

    print(f"\n{'=' * 78}")
    print(f"PROFILE  system={system_name}  pipeline={pipeline}  seed={seed}"
          f"  epochs={cfg.hparams['moeadd']['training_epochs']}")
    print(f"{'=' * 78}")

    profiler = cProfile.Profile()
    wall_start = time.time()
    profiler.enable()
    try:
        search = build_search(cfg, pipeline_kwargs)
    finally:
        profiler.disable()
    wall_total = time.time() - wall_start
    print(f"\n[wall] build_search total: {wall_total:.2f}s")

    out_dir = os.path.join(_THIS_DIR, 'profile_results')
    os.makedirs(out_dir, exist_ok=True)
    prof_path = os.path.join(out_dir, f"{system_name}_{pipeline}_seed{seed}.prof")
    profiler.dump_stats(prof_path)
    print(f"[prof] dumped {prof_path}  (snakeviz {prof_path}  to visualize)")

    stats = pstats.Stats(profiler).strip_dirs()

    print("\n--- top 30 functions by cumulative time ---")
    print(_format_top_n(stats, 'cumulative', 30))

    print("--- top 30 functions by self time (tottime) ---")
    print(_format_top_n(stats, 'tottime', 30))

    print("--- per-operator (.apply) aggregation, sorted by cumtime ---")
    rows = _aggregate_apply_methods(pstats.Stats(profiler))
    header = f"{'cumtime':>10} {'tottime':>10} {'ncalls':>8} {'%wall':>7}  file:lineno"
    print(header)
    print('-' * len(header))
    for r in rows[:25]:
        pct = (r['cumtime'] / wall_total * 100.0) if wall_total > 0 else 0.0
        print(f"{r['cumtime']:>10.2f} {r['tottime']:>10.2f} {r['ncalls']:>8d} {pct:>6.1f}%  {r['key']}")
    print(f"\n[wall] build_search total = {wall_total:.2f}s")
    print("(Note: cumtime overlaps -- outer ops include their suboperators.)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('systems', nargs='*', default=['lv', 'wave'],
                   help="systems to profile (default: lv wave)")
    p.add_argument('--pipeline', default='new', choices=('legacy', 'new'),
                   help="pipeline label (default: new)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=None,
                   help="override moeadd.training_epochs for the run")
    args = p.parse_args(argv)

    for system_name in args.systems:
        profile_system(system_name, pipeline=args.pipeline, seed=args.seed,
                       epochs=args.epochs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
