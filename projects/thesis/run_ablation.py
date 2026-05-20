"""Ablation entry point: same CLI as ``run.py`` but defaults to the six
off-diagonal cells of the 2x2x2 factorial (``wape``, ``instab``, ``reg``,
``wape_instab``, ``wape_reg``, ``instab_reg``).

The 000 (``legacy``) and 111 (``new``) corners are *not* run here -- they
are produced by ``run.py`` and the aggregator reads both label sets from
the same results tree.

Usage:
    python projects/thesis/run_ablation.py <system> [--reps N]
                                                    [--outdir TAG]
                                                    [--no-resume]
"""

from __future__ import annotations

import sys

from run import main as _main  # noqa: E402
from thesis_runner import ABLATION_PIPELINES  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Inject the ablation pipelines unless the caller passed --pipelines explicitly.
    if not any(a == '--pipelines' or a.startswith('--pipelines=') for a in argv):
        argv += ['--pipelines', *ABLATION_PIPELINES]
    return _main(argv)


if __name__ == '__main__':
    sys.exit(main())
