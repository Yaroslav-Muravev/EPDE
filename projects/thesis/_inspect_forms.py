"""Aggregate recurring discovered structural forms across reps."""
import json
import glob
import re
from collections import Counter

BACKSLASH = chr(92)

for sys in ['lv', 'lorenz', 'burgers_inviscid']:
    print(f'=== {sys} ===')
    paths = sorted(
        glob.glob(f'C:/Users/NSSLab/PycharmProjects/EPDE/projects/thesis/'
                  f'results/{sys}/new_rep*.json')
    )
    paths = [p for p in paths if '.history' not in p]
    canon_count = Counter()
    raw_samples = {}
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        if not r.get('discovered_text_per_solution'):
            continue
        for sol in r['discovered_text_per_solution']:
            if not isinstance(sol, list):
                continue
            eqs = [s for s in sol if isinstance(s, str) and '=' in s]
            stripped = []
            for eq in eqs:
                e = eq.strip()
                for ch in ('/', BACKSLASH, '|'):
                    e = e.lstrip(ch)
                e = e.strip()
                # Drop numeric coefs; keep structure.
                e = re.sub(r'-?\d+\.\d+(e[-+]?\d+)?', 'C', e)
                stripped.append(e)
            sig = tuple(sorted(stripped))
            canon_count[sig] += 1
            if sig not in raw_samples:
                raw_samples[sig] = eqs
    for sig, n in canon_count.most_common(8):
        print(f' [{n}x]')
        for e in raw_samples[sig]:
            print(f'     {e}')
        print()
    print()
