"""One-shot post-processor: aggregate .apply cumtime per operator from a .prof file."""
import argparse
import pstats
from collections import defaultdict


def aggregate(prof_path: str, wall: float) -> None:
    stats = pstats.Stats(prof_path)
    per = defaultdict(lambda: {'cumtime': 0.0, 'tottime': 0.0, 'ncalls': 0, 'key': ''})
    for func_key, (cc, nc, tt, ct, _callers) in stats.stats.items():
        fname, lineno, funcname = func_key
        if funcname != 'apply':
            continue
        fn = fname.replace('\\', '/').lower()
        if '/epde/' not in fn:
            continue
        short = fname.replace('\\', '/').split('/epde/')[-1]
        key = f"{short}:{lineno}"
        b = per[key]
        b['cumtime'] += ct
        b['tottime'] += tt
        b['ncalls'] += nc
        b['key'] = key
    rows = sorted(per.values(), key=lambda r: r['cumtime'], reverse=True)
    print(f"{'cumtime':>10} {'tottime':>10} {'ncalls':>10} {'%wall':>7}  file:lineno")
    print('-' * 80)
    for r in rows[:30]:
        pct = r['cumtime'] / wall * 100.0 if wall > 0 else 0.0
        print(f"{r['cumtime']:>10.2f} {r['tottime']:>10.2f} {r['ncalls']:>10d} {pct:>6.1f}%  {r['key']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('prof_path')
    p.add_argument('--wall', type=float, default=0.0)
    args = p.parse_args()
    aggregate(args.prof_path, args.wall)
