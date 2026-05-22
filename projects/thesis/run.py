"""Unified CLI entry for the thesis Section 4.5 main comparison.

Usage:
    python projects/thesis/run.py <system> [--reps N] [--pipelines legacy new]
                                            [--outdir TAG] [--no-resume]
                                            [--seed-base N]

``<system>`` matches one of the YAML files in ``projects/thesis/configs/``
(e.g. ``lv``, ``lorenz``, ``kdv``). The default pipelines are ``legacy`` and
``new``; pass ``--pipelines`` to override (the eight valid labels are
``legacy``, ``new``, and the six off-diagonal ablation labels -- see
``thesis_runner._PIPELINE_SETTINGS``).

``--outdir TAG`` redirects results to ``results/TAG/<system>/`` so a tagged
sweep across multiple systems stays grouped under one folder. ``--outdir
/abs/path`` lands there directly.
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from thesis_runner import (  # noqa: E402
    ABLATION_PIPELINES,
    CONFIGS_DIR,
    PIPELINES,
    _PIPELINE_SETTINGS,
    load_config,
    run_smoke,
)


def _available_systems() -> list:
    if not os.path.isdir(CONFIGS_DIR):
        return []
    # ``defaults.yaml`` is the shared baseline that load_config merges
    # under every per-system config; it's not itself a runnable system.
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(CONFIGS_DIR)
        if f.endswith('.yaml') and f != 'defaults.yaml'
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'system',
        help=f"system name (looked up as configs/<system>.yaml). "
             f"Available: {', '.join(_available_systems()) or '(none yet)'}",
    )
    parser.add_argument('--reps', type=int, default=30,
                        help="reps per pipeline (default: 30)")
    parser.add_argument(
        '--pipelines', nargs='+', default=list(PIPELINES),
        choices=tuple(_PIPELINE_SETTINGS),
        help=f"pipeline labels (default: {' '.join(PIPELINES)})",
    )
    parser.add_argument('--outdir', default=None,
                        help="results tag (lands at results/<tag>/<system>/) "
                             "or absolute path; default reuses cfg's outdir")
    parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                        help="overwrite existing per-rep JSONs instead of skipping")
    parser.add_argument('--seed-base', type=int, default=0,
                        help="seed for rep 0 (rep i uses seed_base + i)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.system)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    run_smoke(
        cfg,
        reps=args.reps,
        pipelines=tuple(args.pipelines),
        seed_base=args.seed_base,
        resume=args.resume,
        outdir=args.outdir,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
