"""Run a single NEW-pipeline rep with loop counters enabled and report.

Sets ``EPDE_LOOP_STATS=1`` in the process environment **before** any
epde import, then runs ``build_search`` for each system (defaults:
lv + wave), prints the loop-stats table, and writes it to
``projects/thesis/profile_results/loop_stats_<system>.txt``.

Usage:
    python projects/thesis/profile_loop_stats.py [system ...] [--epochs N] [--seed N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time


# Enable instrumentation BEFORE any epde import — _loop_stats reads
# the env var at module-load time.
os.environ['EPDE_LOOP_STATS'] = '1'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from epde import _loop_stats  # noqa: E402
from thesis_runner import (  # noqa: E402
    _set_seeds,
    build_search,
    load_config,
    pipeline_settings,
)


def profile_system(system_name: str, pipeline: str = 'new', seed: int = 0,
                   epochs: int | None = None) -> None:
    cfg = load_config(system_name)
    cfg.hparams['moeadd']['early_stop_on_truth'] = False
    if epochs is not None:
        cfg.hparams['moeadd']['training_epochs'] = int(epochs)

    pipeline_kwargs = pipeline_settings(pipeline)
    _set_seeds(seed)

    print(f"\n{'=' * 78}")
    print(f"LOOP-STATS  system={system_name}  pipeline={pipeline}  seed={seed}"
          f"  epochs={cfg.hparams['moeadd']['training_epochs']}")
    print(f"{'=' * 78}")

    _loop_stats.reset()
    t0 = time.time()
    build_search(cfg, pipeline_kwargs)
    wall = time.time() - t0
    print(f"\n[wall] build_search total: {wall:.2f}s\n")

    out_dir = os.path.join(_THIS_DIR, 'profile_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"loop_stats_{system_name}.txt")
    text = _loop_stats.report(path=out_path)
    print(text)
    print(f"\n[saved] {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('systems', nargs='*', default=['lv', 'wave'])
    p.add_argument('--pipeline', default='new', choices=('legacy', 'new'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=5)
    args = p.parse_args(argv)

    for system_name in args.systems:
        profile_system(system_name, pipeline=args.pipeline, seed=args.seed,
                       epochs=args.epochs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
