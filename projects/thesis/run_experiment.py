"""Run a multi-system experiment declared in an experiment YAML.

Usage:
    python projects/thesis/run_experiment.py <experiment>
                                              [--system <name>]
                                              [--reps N]
                                              [--pipelines P [P ...]]
                                              [--no-resume]
                                              [--seed-base N]
                                              [--outdir TAG]

``<experiment>`` is either a bare name (looked up under
``projects/thesis/experiments/<name>.yaml``) or an explicit path. CLI
flags override the corresponding YAML fields for the entire sweep --
useful for smoke (``--reps 1``) or ad-hoc retargeting
(``--outdir test_v2``). ``--system <name>`` restricts the sweep to a
single entry from the YAML's ``systems`` list, useful for resuming a
failed partial run on one GPU.

Errors on one system don't kill the sweep: they're caught, logged,
and the loop continues to the next system. Final summary prints
elapsed time per system + an overall counter.

Layout:
    projects/thesis/experiments/<name>.yaml      <-- this loader
    projects/thesis/configs/<system>.yaml        <-- per-system cfg
    projects/thesis/results/<outdir>/<system>/   <-- per-rep JSON output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from thesis_runner import (  # noqa: E402
    EXPERIMENTS_DIR,
    _PIPELINE_SETTINGS,
    _load_yaml,
    load_config,
    run_smoke,
)


_REQUIRED_KEYS = ('name', 'systems', 'pipelines', 'reps')


def _load_experiment(name_or_path: str) -> dict:
    """Resolve an experiment name to its YAML payload + validate required keys."""
    yaml_path = (
        name_or_path
        if os.path.sep in name_or_path or name_or_path.endswith('.yaml')
        else os.path.join(EXPERIMENTS_DIR, f'{name_or_path}.yaml')
    )
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"experiment not found: {yaml_path}")
    d = _load_yaml(yaml_path)
    missing = [k for k in _REQUIRED_KEYS if k not in d]
    if missing:
        raise ValueError(f"{yaml_path}: missing required key(s): {missing}")
    bad_pipelines = [p for p in d['pipelines'] if p not in _PIPELINE_SETTINGS]
    if bad_pipelines:
        raise ValueError(
            f"{yaml_path}: unknown pipelines {bad_pipelines}; expected "
            f"a subset of {tuple(_PIPELINE_SETTINGS)}"
        )
    return d


def _available_experiments() -> list:
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(EXPERIMENTS_DIR)
        if f.endswith('.yaml')
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'experiment',
        help=f"experiment name (looked up as experiments/<experiment>.yaml). "
             f"Available: {', '.join(_available_experiments()) or '(none yet)'}",
    )
    parser.add_argument('--system', default=None,
                        help="restrict the sweep to a single system from the experiment's "
                             "systems list (useful for resuming or splitting across GPUs)")
    parser.add_argument('--reps', type=int, default=None,
                        help="override the experiment's reps (default: keep YAML value)")
    parser.add_argument('--pipelines', nargs='+', default=None,
                        choices=tuple(_PIPELINE_SETTINGS),
                        help="override the experiment's pipelines list")
    parser.add_argument('--outdir', default=None,
                        help="override the experiment's outdir tag (relative -> "
                             "results/<tag>/<system>/, absolute -> tag/)")
    parser.add_argument('--no-resume', dest='resume', action='store_false', default=None,
                        help="overwrite existing per-rep JSONs instead of skipping")
    parser.add_argument('--seed-base', type=int, default=None,
                        help="override the experiment's seed_base")
    args = parser.parse_args(argv)

    try:
        exp = _load_experiment(args.experiment)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # CLI overrides win where supplied; otherwise use YAML defaults.
    systems = exp['systems']
    if args.system is not None:
        if args.system not in systems:
            parser.error(
                f"--system {args.system!r} is not in the experiment's systems list "
                f"({systems})"
            )
        systems = [args.system]
    pipelines = tuple(args.pipelines) if args.pipelines is not None else tuple(exp['pipelines'])
    reps = args.reps if args.reps is not None else int(exp['reps'])
    seed_base = args.seed_base if args.seed_base is not None else int(exp.get('seed_base', 0))
    resume = args.resume if args.resume is not None else bool(exp.get('resume', True))
    outdir = args.outdir if args.outdir is not None else exp.get('outdir', exp['name'])

    print(f"========== experiment={exp['name']} ==========")
    print(f"  systems   = {systems}")
    print(f"  pipelines = {pipelines}")
    print(f"  reps      = {reps}")
    print(f"  seed_base = {seed_base}")
    print(f"  resume    = {resume}")
    print(f"  outdir    = {outdir}")
    print()

    timings = {}
    failures = {}
    for system_name in systems:
        t0 = time.time()
        try:
            cfg = load_config(system_name)
            run_smoke(
                cfg,
                reps=reps,
                pipelines=pipelines,
                seed_base=seed_base,
                resume=resume,
                outdir=outdir,
            )
        except Exception as exc:  # don't let one bad system kill the sweep
            failures[system_name] = repr(exc)
            print(f"[run_experiment] system={system_name!r} FAILED: {exc!r}")
            traceback.print_exc()
        timings[system_name] = time.time() - t0

    print()
    print("========== experiment summary ==========")
    for system_name in systems:
        status = 'FAIL' if system_name in failures else 'OK'
        print(f"  {system_name:24} {status:4}  {timings[system_name]:.1f}s")
    if failures:
        print(f"\n{len(failures)} system(s) failed: {sorted(failures)}")
        return 1
    print(f"\nall {len(systems)} systems OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
