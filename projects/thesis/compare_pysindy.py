"""Compare PySINDy single-shot stats against EPDE-new 30-rep aggregate.

For each system, compute two extra columns not in ``thesis_aggregate``:

1. **Mean coefficient relative error** across reps with
   ``structural_success=True``. Pairs discovered terms with truth terms
   structurally (same canonical form used by ``thesis_metrics``), then
   normalises both equations so the truth target term carries
   coefficient 1, and sums ``|c_disc - c_truth| / |c_truth|`` over all
   terms appearing in either equation. Mirrors PySINDy's "sum of
   relative errors" metric so the two pipelines can be compared on the
   same scale.

2. **Search-space cardinality** (single-factor universe + estimated
   1-to-K-factor term universe). Built by loading each system's config
   + data, instantiating the EPDE token pool, walking every
   ``TokenFamily``'s tokens and parameter-bucket combinations, and
   summing.

Output: markdown table to stdout + JSON sidecar.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import yaml  # type: ignore  # noqa: E402

from thesis_metrics import (  # noqa: E402
    _FACTOR_RE, _PARAM_RE, _round_param, _GRID_COORD_NAME_RE,
)


def _parse_factor_with_coef(text: str):
    """Same as ``thesis_metrics._parse_factor`` but exposed locally."""
    m = _FACTOR_RE.search(text)
    if m is None:
        return None
    name = m.group(1)
    params = {}
    for pm in _PARAM_RE.finditer(m.group(2)):
        params[pm.group(1)] = _round_param(pm.group(2))
    if _GRID_COORD_NAME_RE.match(name):
        name = 'x'
    return (name, frozenset(params.items()))


def _parse_term_with_coef(text: str):
    """Parse ``c * f1{...} * f2{...}`` into ``(coef, frozenset_of_factors)``.

    Pure-constant terms collapse to ``(coef, None)``; the caller must
    treat them as intercepts (currently dropped).
    """
    pieces = [p.strip() for p in text.split('*')]
    factors = []
    coef = 1.0
    coef_seen = False
    for piece in pieces:
        if not piece:
            continue
        factor = _parse_factor_with_coef(piece)
        if factor is None:
            try:
                val = float(piece)
                coef *= val
                coef_seen = True
            except ValueError:
                continue
        else:
            factors.append(factor)
    if not factors:
        return (coef if coef_seen else 0.0, None)
    if not coef_seen:
        coef = 1.0
    return (coef, frozenset(factors))


def _parse_equation_as_coef_vector(eq_text: str):
    """Parse ``LHS = RHS`` and return ``({term_sig: coef}, rhs_term_sig)``.

    The returned dict represents the algebraic ``LHS - RHS = 0`` form so
    every term carries a signed coefficient. ``rhs_term_sig`` is the
    target term's structural signature.

    Strips EPDE's multi-equation display prefixes (``/``, ``\\``, ``|``)
    that ``SoEq.text_form`` prepends per row -- these otherwise corrupt
    the leading coefficient when the term parser tries to split by ``*``
    and parse ``/ -19.88`` as a single numeric piece (which it isn't,
    yielding coef = 1.0 silently and undercounting the term's magnitude).
    """
    if '=' not in eq_text:
        return None, None
    eq_text = eq_text.strip().lstrip('/').lstrip('\\').lstrip('|').strip()
    left, right = eq_text.split('=', 1)
    coefs = {}
    for piece in left.split('+'):
        c, sig = _parse_term_with_coef(piece)
        if sig is None:
            continue
        coefs[sig] = coefs.get(sig, 0.0) + c
    target_coef, target_sig = _parse_term_with_coef(right)
    if target_sig is None:
        return coefs, None
    # Move the RHS target to LHS with sign flip.
    coefs[target_sig] = coefs.get(target_sig, 0.0) - target_coef
    return coefs, target_sig


def _coef_relerr(disc_text: str, truth_text: str) -> Optional[float]:
    """Return PySINDy-style sum of per-term relative coefficient errors.

    Both equations are converted to ``coef^T term = 0`` form, then
    normalised so the truth target term's coefficient is 1. The error
    is ``sum_i |c_disc_norm[i] - c_truth_norm[i]| / max(|c_truth_norm[i]|, 1e-10)``
    over all term signatures appearing in either equation. Returns
    ``None`` if the truth's target term isn't present in discovered
    (can't normalise to the same pivot).
    """
    disc_coefs, _ = _parse_equation_as_coef_vector(disc_text)
    truth_coefs, truth_target = _parse_equation_as_coef_vector(truth_text)
    if disc_coefs is None or truth_coefs is None or truth_target is None:
        return None
    truth_pivot = truth_coefs.get(truth_target)
    disc_pivot = disc_coefs.get(truth_target)
    if truth_pivot is None or disc_pivot is None or abs(disc_pivot) < 1e-12:
        return None
    truth_norm = {k: v / truth_pivot for k, v in truth_coefs.items()}
    disc_norm = {k: v / disc_pivot for k, v in disc_coefs.items()}
    all_sigs = set(truth_norm) | set(disc_norm)
    total = 0.0
    for sig in all_sigs:
        t = truth_norm.get(sig, 0.0)
        d = disc_norm.get(sig, 0.0)
        denom = max(abs(t), 1e-10)
        total += abs(d - t) / denom
    return total


def _rep_relerr_against_truth(disc_eqs: list, truth_eqs: list) -> Optional[float]:
    """Sum of per-truth-equation best-match rel-err for one (rep, truth-form).

    Pairs each truth equation with the discovered equation that
    minimises ``_coef_relerr``. Returns the sum across truth equations,
    or ``None`` if no truth equation could be matched (e.g. discovered
    has none of the same target-term structures as truth).
    """
    rep_errs: list[float] = []
    for t_eq in truth_eqs:
        best = None
        for d_eq in disc_eqs:
            e = _coef_relerr(d_eq, t_eq)
            if e is None:
                continue
            if best is None or e < best:
                best = e
        if best is not None:
            rep_errs.append(best)
    if not rep_errs:
        return None
    return sum(rep_errs)


def _system_coef_relerr(system: str, rerun_root: str,
                         truth_forms: list[list[str]]
                         ) -> tuple[Optional[float], int]:
    """Mean coefficient rel-err across structurally-successful reps.

    ``truth_forms`` is the list of all accepted analytical forms for the
    system: ``[primary truth_equations, alt_1, alt_2, ...]``. For each
    successful rep we compute the per-form rep rel-err (greedy pairing
    of discovered to truth equations) and keep the MINIMUM across
    forms, so a rep credited via a ``truth_alternatives`` form (e.g.
    burgers_inviscid's similarity identity) reports coefficient quality
    against that alternative.
    """
    paths = sorted(glob.glob(os.path.join(rerun_root, system, 'new_rep*.json')))
    paths = [p for p in paths if not p.endswith('.history.json')]
    per_rep_errs: list[float] = []
    for p in paths:
        try:
            with open(p, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not rec.get('structural_success'):
            continue
        disc_text = rec.get('discovered_text', [])
        if not isinstance(disc_text, list) or not disc_text:
            continue
        disc_eqs = [s for s in disc_text if isinstance(s, str) and '=' in s]
        if not disc_eqs:
            continue
        best_form_err = None
        for truth_eqs in truth_forms:
            err = _rep_relerr_against_truth(disc_eqs, truth_eqs)
            if err is None:
                continue
            if best_form_err is None or err < best_form_err:
                best_form_err = err
        if best_form_err is not None:
            per_rep_errs.append(best_form_err)
    if not per_rep_errs:
        return None, 0
    return sum(per_rep_errs) / len(per_rep_errs), len(per_rep_errs)


def _enumerate_param_buckets(token_params: dict, equality_ranges: dict) -> int:
    """Count distinct param-bucket combinations for one token.

    For each param: count = bounds-range / equality_range + 1 if
    equality_range > 0; else 1 (single value).
    """
    count = 1
    for name, bounds in token_params.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        eq = float(equality_ranges.get(name, 0))
        if eq <= 0 or hi == lo:
            buckets = 1
        else:
            buckets = max(1, int(round((hi - lo) / eq)) + 1)
        count *= buckets
    return count


def _system_universe_size(system: str) -> Optional[tuple[int, int, list[str]]]:
    """Estimate the factor-signature universe + term universe analytically
    from the system's YAML config.

    Computed families (matches what EPDE's ``EpdeSearch.fit`` builds
    internally plus what ``_build_token_pool`` adds):

    * Variable tokens: one per state variable, with ``power``
      buckets 1..``data_fun_pow``.
    * Derivative tokens: per (variable, axis, order) up to
      ``max_deriv_order`` per axis, with ``power`` buckets
      1..``deriv_fun_pow``.
    * Grid token: one token ``x`` with ``dim`` 0..``dim_count - 1`` and
      ``power`` buckets 1..``grid_tokens.max_power``.
    * TrigonometricTokens / others declared under
      ``additional_tokens``: query the constructed family for token
      count + parameter buckets.

    Returns ``(factor_universe, term_universe_up_to_max_k, family_descr)``.
    Term universe = ``sum_{k=1..max_k} C(factor_universe, k)`` which is
    the upper bound (ignores ``filter_powers`` collapse and per-factor
    occupancy caps).
    """
    try:
        from thesis_runner import load_config, _build_token_pool
    except Exception:
        return None
    try:
        cfg = load_config(system)
        coords, _, variable_names, dim = cfg.load_data()
    except Exception as exc:
        print(f'  [warn] {system}: load_data failed: {exc!r}', file=sys.stderr)
        return None

    n_vars = len(variable_names)
    fit = cfg.hparams['fit']
    data_fun_pow = int(fit['data_fun_pow'])
    deriv_fun_pow = int(fit['deriv_fun_pow'])
    max_deriv_order = fit.get('max_deriv_order')
    if max_deriv_order is None:
        # ODE (dim=0) -> (2,); 1+1D PDE -> (2, 4); 2+1D PDE -> (2, 4, 4).
        max_deriv_order = (2,) + (4,) * dim
    else:
        max_deriv_order = tuple(max_deriv_order)

    family_descr = []
    factor_universe = 0

    # 1. Variable tokens (u, v, ...): one per variable, power buckets 1..data_fun_pow.
    var_sigs = n_vars * data_fun_pow
    factor_universe += var_sigs
    family_descr.append(f"vars({n_vars}x{data_fun_pow}p={var_sigs})")

    # 2. Derivative tokens: per (variable, axis, order). max_deriv_order[axis]
    #    gives the highest derivative order along that axis. Each order gets
    #    a distinct token name, and each token has power buckets 1..deriv_fun_pow.
    deriv_tokens = 0
    for axis_order in max_deriv_order:
        deriv_tokens += n_vars * int(axis_order)
    deriv_sigs = deriv_tokens * deriv_fun_pow
    factor_universe += deriv_sigs
    family_descr.append(f"derivs({deriv_tokens}x{deriv_fun_pow}p={deriv_sigs})")

    # 3. Grid token (single ``x`` family after consolidation): dim buckets
    #    0..dim_count and power buckets 1..max_power.
    gt = cfg.hparams['grid_tokens']
    grid_power = int(gt['max_power'])
    grid_dim_buckets = dim + 1  # axes 0..dim inclusive
    grid_sigs = grid_power * grid_dim_buckets
    factor_universe += grid_sigs
    family_descr.append(f"grid({grid_dim_buckets}x{grid_power}p={grid_sigs})")

    # 4. additional_tokens from YAML (TrigonometricTokens etc.) — pull
    #    cardinality directly from the constructed family.
    try:
        pool_objs = _build_token_pool(cfg, coords, dim)
    except Exception:
        pool_objs = []
    for pt in pool_objs:
        tf = getattr(pt, '_token_family', None)
        if tf is None:
            continue
        # Skip the grid family — already counted above. Identified by
        # token_type == 'grids' (set in GridTokens.__init__).
        if getattr(tf, 'ftype', '') == 'grids':
            continue
        tokens = list(getattr(tf, 'tokens', []) or [])
        token_params = getattr(tf, 'token_params', {}) or {}
        equality_ranges = getattr(tf, 'equality_ranges', {}) or {}
        per_token = _enumerate_param_buckets(token_params, equality_ranges)
        size = len(tokens) * per_token
        factor_universe += size
        family_descr.append(f"{tf.ftype}({len(tokens)}x{per_token}={size})")

    fma = fit['equation_factors_max_number']
    max_k = max(fma['factors_num']) if fma['factors_num'] else 1

    # Term universe: sum of C(factor_universe, k) for k=1..max_k.
    term_universe = 0
    n = factor_universe
    for k in range(1, max_k + 1):
        if k > n:
            break
        term_universe += math.comb(n, k)
    return factor_universe, term_universe, family_descr


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rerun-root',
                   default=os.path.join(_THIS_DIR, 'results'),
                   help='Root containing per-system rep JSONs.')
    p.add_argument('--configs-dir',
                   default=os.path.join(_THIS_DIR, 'configs'),
                   help='YAML configs root.')
    p.add_argument('--systems',
                   default='ac,burgers_inviscid,burgers_viscous,kdv,'
                           'kdv_cossin,ks,lorenz,lv,ode,pde_compound,'
                           'pde_divide,vdp,wave',
                   help='Comma-separated system list.')
    p.add_argument('--skip-universe', action='store_true',
                   help='Skip data-loading step (universe-size columns blank).')
    p.add_argument('--out',
                   default=os.path.join(_THIS_DIR, 'pysindy_comparison.json'),
                   help='JSON sidecar with the parsed data.')
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(',') if s.strip()]
    out = {}
    for system in systems:
        cfg_path = os.path.join(args.configs_dir, f'{system}.yaml')
        if not os.path.exists(cfg_path):
            print(f'  [skip] {system}: no config', file=sys.stderr)
            continue
        with open(cfg_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
        # All accepted analytical forms: primary ``truth_equations``
        # plus every entry in ``truth_alternatives``. Coefficient
        # rel-err is computed against each and the best (lowest) per
        # rep wins, so a rep credited structurally via an alternative
        # (e.g. burgers_inviscid's similarity identity) reports its
        # coefficient quality against that alternative.
        truth_forms = []
        primary = cfg.get('truth_equations') or []
        if primary:
            truth_forms.append(list(primary))
        for alt in (cfg.get('truth_alternatives') or []):
            if alt:
                truth_forms.append(list(alt))

        mean_relerr, n_success = _system_coef_relerr(
            system, args.rerun_root, truth_forms)

        univ = None
        if not args.skip_universe:
            print(f'  [enum] {system}: building token pool...', file=sys.stderr)
            try:
                univ = _system_universe_size(system)
            except Exception:
                traceback.print_exc()
                univ = None

        rec = {
            'system': system,
            'coef_relerr_mean': mean_relerr,
            'coef_relerr_n_success': n_success,
        }
        if univ is not None:
            factor_n, term_n, descr = univ
            rec['factor_universe'] = factor_n
            rec['term_universe'] = term_n
            rec['family_descr'] = descr
        out[system] = rec
        print(f'  {system}: relerr={mean_relerr}, n_success={n_success}, '
              f'univ={univ[:2] if univ else None}', file=sys.stderr)

    # Markdown table
    print()
    print('| System | EPDE coef relerr (mean over success reps) | n_success | '
          'EPDE factor universe | EPDE term universe (k<=max) |')
    print('|---|---|---|---|---|')
    for system in systems:
        rec = out.get(system, {})
        relerr = rec.get('coef_relerr_mean')
        n = rec.get('coef_relerr_n_success', 0)
        fu = rec.get('factor_universe', '-')
        tu = rec.get('term_universe', '-')
        relerr_s = f'{relerr:.3e}' if relerr is not None else '-'
        if isinstance(tu, int) and tu >= 10000:
            tu_s = f'{tu:,}'
        else:
            tu_s = str(tu)
        print(f'| {system} | {relerr_s} | {n} | {fu} | {tu_s} |')

    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f'\nWrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
