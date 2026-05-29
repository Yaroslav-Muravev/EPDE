"""Per-system candidate-history plot, one figure per pipeline.

Combines the all-candidates-throughout-search view of
``plot_objectives_density.py`` (every candidate from every Pareto
level across every epoch, read from the sidecar
``<rep>.history.json``) with the truth-matching overlay used in
``plot_pareto_correct.py``.

LEGACY and NEW live in different objective spaces:
  LEGACY: (L2 discrepancy, complexity) — set by
          ``use_legacy_multiobjective_function`` in main_structures.py.
  NEW:    (WAPE discrepancy, instability) — set by
          ``use_new_multiobjective_function``.
Each pipeline gets its own figure so the two objective spaces don't
have to share axes. For each system we pick the first seed where
both pipelines have a history sidecar (when available), preferring
seeds where NEW matched truth so the star overlay is meaningful.

Output: per-system PNGs split by pipeline at
``projects/thesis/figures/history_match_<pipeline>_<system>.png``
(one for LEGACY, one for NEW) plus two grid montages at
``projects/thesis/figures/history_match_grid_<pipeline>.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Seaborn theme: clean ticks, soft grid, sans-serif. ``whitegrid`` keeps
# the log-log gridlines visible against the candidate cloud while
# softening the surrounding chrome.
sns.set_theme(style='whitegrid', context='notebook',
              rc={'axes.spines.right': False, 'axes.spines.top': False})
# Bright, perceptually-uniform candidate colour (Crest mid-tone) for NEW
# pipeline; warm flare mid-tone for LEGACY so the two clouds separate
# without becoming muddy when they overlap.
_CLOUD_COLOR_NEW = sns.color_palette('crest', n_colors=5)[2]
_CLOUD_COLOR_LEGACY = sns.color_palette('flare', n_colors=5)[2]
# High-contrast truth markers: Set1 red star for NEW, Set1 blue diamond
# for LEGACY.
_MATCH_COLOR_NEW = sns.color_palette('Set1', n_colors=9)[0]
_MATCH_COLOR_LEGACY = sns.color_palette('Set1', n_colors=9)[1]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
_FIG_DIR = os.path.join(_PROJECT_DIR, 'figures')

SYSTEM_ORDER = [
    'ode', 'vdp', 'lorenz', 'lv',
    'ac', 'burgers_inviscid', 'burgers_viscous',
    'kdv', 'kdv_cossin', 'ks',
    'wave', 'pde_compound', 'pde_divide',
    'ns',
]


def _load_history(rep_dir: str, history_basename: str):
    hist_path = os.path.join(rep_dir, history_basename)
    try:
        with open(hist_path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        return payload.get('candidate_history')
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"history load failed: {hist_path}: {exc!r}")
        return None


def _iter_reps_with_history(system: str, pipeline: str):
    """Yield (rec, rep_path) for every rep of ``pipeline`` whose sidecar
    history exists, in seed-sorted order."""
    pattern = os.path.join(_RESULTS_DIR, system, f'{pipeline}_rep*.json')
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        hist = rec.get('history_path')
        if not hist:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(path), hist)):
            continue
        yield rec, path


def _find_first_success(system: str, pipeline: str = 'new'):
    """First rep of ``pipeline`` with structural_success AND a sidecar."""
    for rec, path in _iter_reps_with_history(system, pipeline):
        if rec.get('structural_success'):
            return rec, path
    return None, None


def _find_first_with_history(system: str, pipeline: str = 'new'):
    """First rep of ``pipeline`` with a sidecar (success not required)."""
    for rec, path in _iter_reps_with_history(system, pipeline):
        return rec, path
    return None, None


def _freeze(obj):
    """Recursively convert lists -> tuples so the result is hashable."""
    if isinstance(obj, list):
        return tuple(_freeze(x) for x in obj)
    return obj


def _looks_like_factor(item) -> bool:
    """``[name_str, params_list]`` shape -- one factor of one term."""
    return (isinstance(item, list) and len(item) == 2
            and isinstance(item[0], str))


def _looks_like_term(item) -> bool:
    """``[factor, factor, ...]`` shape -- one term (list of factors)."""
    return (isinstance(item, list) and len(item) > 0
            and _looks_like_factor(item[0]))


def _term_to_frozenset(term_factors) -> frozenset:
    """Convert one ``[factor, factor, ...]`` term into a frozenset of
    ``(name, params_frozenset)`` factor tuples. Uses ``frozenset`` for
    params so the result compares equal to ``thesis_metrics`` native
    output (also frozenset-of-frozenset-of-tuple-of-tuple). Returns the
    empty frozenset if no factor parses."""
    factors = []
    for factor in term_factors:
        # Some product tokens (e.g. ``cos(t)sin(x)`` on kdv_cossin) come
        # in JSON wrapped as ``[[name, params]]``; unwrap one level.
        if (isinstance(factor, list) and len(factor) == 1
                and _looks_like_factor(factor[0])):
            factor = factor[0]
        if not _looks_like_factor(factor):
            continue
        name = factor[0]
        params = frozenset(
            tuple(_freeze(p)) for p in (factor[1] or ())
        )
        factors.append((name, params))
    return frozenset(factors)


def _token_eq_to_frozenset(eq_tokens) -> frozenset:
    """Convert a per-equation token list into a target-side-independent
    frozenset of term-frozensets so two equations can be compared by
    ``==`` regardless of which side is the target.

    Handles two stored shapes:
      * **Flat term list** (the documented canonical form, used by
        e.g. Lorenz): ``[term, term, ...]`` -- target already merged
        into the term set by the runner.
      * **Heterogeneous ``[target_term, rhs_term_list]``** shape (seen
        for kdv_cossin: target/RHS preserved as a 2-element list with
        a nested RHS term-list). Detected and flattened so the two
        shapes compare equal when the underlying equation is the same.

    Param values are recursively frozen so nested-list params don't
    break ``frozenset`` hashing.
    """
    terms = set()
    for top in eq_tokens or ():
        if not isinstance(top, list) or not top:
            continue
        if _looks_like_term(top):
            ts = _term_to_frozenset(top)
            if ts:
                terms.add(ts)
        elif _looks_like_term(top[0]) or (
            isinstance(top[0], list) and top[0]
            and _looks_like_term(top[0][0])
        ):
            # ``top`` is a list of TERMS, not a single term. Flatten.
            for sub in top:
                if _looks_like_term(sub):
                    ts = _term_to_frozenset(sub)
                    if ts:
                        terms.add(ts)
    return frozenset(terms)


def _matched_equation_count(disc_sol_tokens, truth_tokens) -> int:
    """Max count of equation pairings where the discovered equation
    matches a truth equation exactly (set-level), over all bipartite
    pairings. Brute-forces permutations -- safe since len(truth) <= 3
    on all systems in this study."""
    from itertools import permutations
    truth_eqs = [_token_eq_to_frozenset(eq) for eq in (truth_tokens or ())]
    disc_eqs = [_token_eq_to_frozenset(eq) for eq in (disc_sol_tokens or ())]
    if not truth_eqs or not disc_eqs:
        return 0
    n = max(len(disc_eqs), len(truth_eqs))
    cap = min(len(disc_eqs), len(truth_eqs))
    best = 0
    for perm in permutations(range(n)):
        c = 0
        for i in range(n):
            if i >= len(disc_eqs) or perm[i] >= len(truth_eqs):
                continue
            if disc_eqs[i] == truth_eqs[perm[i]]:
                c += 1
        if c > best:
            best = c
            if best == cap:
                return best
    return best


def _rec_best_eq_match_count(rec: dict) -> int:
    """Max ``_matched_equation_count`` across the rec's Pareto-0 solutions.

    Used as a tiebreaker so coupled-system figures (LV / Lorenz / NS)
    pick the seed where the discovered system got the most equations
    right -- even on systems where no rep is a full structural success.
    Compares against the primary ``truth_tokens`` stored on the rec;
    alternative-form matches still get credit through the existing
    ``structural_success`` / ``hamming`` fields.
    """
    truth = rec.get('truth_tokens')
    sols = rec.get('discovered_tokens_per_solution') or []
    best = 0
    for sol in sols:
        c = _matched_equation_count(sol, truth)
        if c > best:
            best = c
    return best


def _seed_quality(rec: dict) -> tuple:
    """Sort key for picking the most-illustrative seed: most truth-matched
    equations first, then full structural success, then lowest Hamming."""
    n_match = _rec_best_eq_match_count(rec)
    success = 1 if rec.get('structural_success') else 0
    ham = rec.get('hamming')
    if ham is None or (isinstance(ham, float) and ham != ham):
        ham = float('inf')
    seed = rec.get('seed', 0) or 0
    # Negate counts so ``min`` picks the best.
    return (-n_match, -success, ham, seed)


def _find_paired_with_history(system: str):
    """Return ((legacy_rec, legacy_path), (new_rec, new_path)) for a single
    seed where BOTH pipelines have a history sidecar. Prefers seeds where
    NEW got the most truth-matched equations (matters for coupled systems
    LV / Lorenz / NS where partial system match is common); falls back to
    lowest seed otherwise. Either side may be None if that pipeline never
    produced a sidecar for the system.
    """
    legacy_by_seed = {rec.get('seed'): (rec, path)
                      for rec, path in _iter_reps_with_history(system, 'legacy')}
    new_by_seed = {rec.get('seed'): (rec, path)
                   for rec, path in _iter_reps_with_history(system, 'new')}
    shared = sorted(set(legacy_by_seed) & set(new_by_seed),
                    key=lambda s: (s is None, s))
    # Preferred: shared seed where NEW had the most truth-matched
    # equations (e.g. 2/3 on Lorenz even if structural_success=False).
    if shared:
        ranked = sorted(shared,
                        key=lambda s: _seed_quality(new_by_seed[s][0]))
        best = ranked[0]
        return legacy_by_seed[best], new_by_seed[best]
    # No shared seed -- pick each side independently using the same
    # quality ordering so the figure still shows whichever cloud(s)
    # exist with the most informative seed per pipeline.
    legacy = (None, None)
    if legacy_by_seed:
        seed = min(legacy_by_seed, key=lambda s: _seed_quality(legacy_by_seed[s][0]))
        legacy = legacy_by_seed[seed]
    new = (None, None)
    if new_by_seed:
        seed = min(new_by_seed, key=lambda s: _seed_quality(new_by_seed[s][0]))
        new = new_by_seed[seed]
    return legacy, new


def _gather_history_points(rec: dict, rep_path: str) -> np.ndarray:
    """Return the SET of unique candidate objective vectors seen across
    the whole search (not the time-stacked history).

    MOEA/D writes the full population once per epoch into ``_hist``, so
    a candidate that survives K epochs would contribute K identical
    rows. Deduplicating collapses those to a single point and removes
    the alpha-stacking artefact (some dots darker than others) that
    otherwise misled readers into thinking the population was non-
    uniform. Each retained point corresponds to a distinct objective
    vector that MOEA/D produced at some point in the search.
    """
    rep_dir = os.path.dirname(rep_path)
    hist_basename = rec.get('history_path')
    if not hist_basename:
        return np.empty((0, 0))
    candidate_history = _load_history(rep_dir, hist_basename)
    if not candidate_history:
        return np.empty((0, 0))
    chunks = []
    for snap in candidate_history:
        if not snap:
            continue
        arr = np.asarray(snap, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            continue
        chunks.append(arr)
    if not chunks:
        return np.empty((0, 0))
    arr = np.vstack(chunks)
    # Round to 6 sig figs so floating-point jitter doesn't dodge dedup.
    _, idx = np.unique(np.round(arr, 6), axis=0, return_index=True)
    # ``np.unique`` returns lex-sorted indices; restoring original order
    # keeps the cloud's first-appearance temporal hint without
    # duplicating points.
    return arr[sorted(idx)]


_CONFIGS_DIR = os.path.join(_PROJECT_DIR, 'configs')
_TRUTH_EQS_CACHE = {}

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)


def _load_truth_equation_canons(system: str) -> list:
    """Return a list of canonical equation-frozensets covering the
    primary truth AND every ``truth_alternatives`` form from the YAML.

    Each entry is one INDIVIDUAL equation (one frozenset of
    term-frozensets) -- not a system. This lets the per-equation
    matcher credit a discovered equation that matches the
    ``u = x*du/dx`` similarity identity (an alternative on
    burgers_inviscid) even when the primary ``du/dt = -u*du/dx`` form
    didn't land.

    Falls back gracefully (empty list) if the YAML is absent or has
    no truth declared.
    """
    if system in _TRUTH_EQS_CACHE:
        return _TRUTH_EQS_CACHE[system]
    path = os.path.join(_CONFIGS_DIR, f'{system}.yaml')
    if not os.path.exists(path):
        _TRUTH_EQS_CACHE[system] = []
        return []
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
    except Exception:
        _TRUTH_EQS_CACHE[system] = []
        return []
    from thesis_metrics import canonical_tokens
    eq_canons = []
    primary = list(cfg.get('truth_equations') or [])
    for eq_text in primary:
        ct = canonical_tokens([eq_text])
        for eq in ct:
            eq_canons.append(eq)
    for alt in (cfg.get('truth_alternatives') or ()):
        for eq_text in (alt or ()):
            ct = canonical_tokens([eq_text])
            for eq in ct:
                eq_canons.append(eq)
    _TRUTH_EQS_CACHE[system] = eq_canons
    return eq_canons


def _final_match_points(rec: dict) -> dict:
    """Per-equation truth-match coordinates for a single rep.

    Returns a dict ``{eq_idx: ndarray(n_match_sols_at_eq_idx, 2)}`` where
    the ``eq_idx`` panel gets a row for each Pareto-0 solution whose
    discovered equation at position ``eq_idx`` matches ANY truth or
    alternative-form equation (loaded from the system's YAML at plot
    time, so reps that match e.g. the ``u = x*du/dx`` similarity form
    on burgers_inviscid are credited too). This decomposes the previous
    all-or-nothing full-system match into per-equation stars so coupled
    systems show stars on equations they got right even without full
    structural success.

    The columns are ``(discrepancy_eq, complexity_eq)`` — taken from
    the solution's ``objectives`` slice for that equation.
    """
    system = rec.get('system')
    truth_eqs = _load_truth_equation_canons(system) if system else []
    if not truth_eqs:
        # Fall back to whatever the rep stored under truth_tokens.
        truth_eqs = [_token_eq_to_frozenset(eq)
                     for eq in (rec.get('truth_tokens') or ())]
    if not truth_eqs:
        return {}
    sols_tokens = rec.get('discovered_tokens_per_solution') or []
    sols_objs = rec.get('objectives_per_solution') or []
    per_eq = {}
    for sol_idx, sol_tokens in enumerate(sols_tokens):
        if sol_idx >= len(sols_objs):
            break
        obj = sols_objs[sol_idx]
        if obj is None:
            continue
        arr = np.asarray(obj, dtype=float).reshape(-1)
        if arr.size == 0 or arr.size % 2 != 0:
            continue
        n_eq_in_obj = arr.size // 2
        for eq_idx, eq_tokens in enumerate(sol_tokens):
            if eq_idx >= n_eq_in_obj:
                break
            disc_canon = _token_eq_to_frozenset(eq_tokens)
            if any(disc_canon == t for t in truth_eqs):
                per_eq.setdefault(eq_idx, []).append(
                    (arr[2 * eq_idx], arr[2 * eq_idx + 1])
                )
    return {k: np.asarray(v, dtype=float) for k, v in per_eq.items()}


def _cohort_match_points(system: str, pipeline: str = 'new') -> dict:
    """Aggregate per-equation truth-match coordinates from EVERY rep of
    ``system`` for ``pipeline``. Used as a fallback when the seed
    providing the history cloud has no per-equation matches of its own
    -- the truth coordinates still live in the cohort and are
    meaningful to overlay because all reps share the system's
    objective-space scale within a pipeline.

    Same dict shape as :func:`_final_match_points`.
    """
    pattern = os.path.join(_RESULTS_DIR, system, f'{pipeline}_rep*.json')
    merged = {}
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        per_eq = _final_match_points(rec)
        for k, v in per_eq.items():
            if k in merged:
                merged[k] = np.vstack([merged[k], v])
            else:
                merged[k] = v
    return merged


def _pipeline_payload(system: str, rec, rep_path):
    """Bundle the cloud + per-equation matches + provenance flag for one
    pipeline. Returns None if no history is available.
    """
    if rec is None or rep_path is None:
        return None
    history = _gather_history_points(rec, rep_path)
    if history.size == 0:
        return None
    matches = _final_match_points(rec)
    matches_from_cohort = False
    if not matches:
        pipeline = rec.get('pipeline', 'new')
        cohort = _cohort_match_points(system, pipeline=pipeline)
        if cohort:
            matches = cohort
            matches_from_cohort = True
    return {
        'rec': rec,
        'rep_path': rep_path,
        'history': history,
        'matches': matches,
        'matches_from_cohort': matches_from_cohort,
    }


# Per-pipeline plotting style: cloud colour, truth-match colour, marker.
_PIPELINE_STYLES = {
    'legacy': dict(cloud=_CLOUD_COLOR_LEGACY, match=_MATCH_COLOR_LEGACY,
                   marker='D', size=70,
                   y_label='complexity', x_label='discrepancy (L2)'),
    'new':    dict(cloud=_CLOUD_COLOR_NEW,    match=_MATCH_COLOR_NEW,
                   marker='*', size=220,
                   y_label='instability', x_label='discrepancy (WAPE)'),
}


def _draw_single(ax, payload, eq_idx: int, pipeline: str) -> int:
    """Draw one pipeline's cloud + per-equation truth-match into ``ax``.
    Returns the candidate count drawn. Stars appear on this panel only
    if some Pareto-0 solution's equation at position ``eq_idx`` matched
    a truth equation exactly -- so coupled systems get individual stars
    on the equations they got right even without full structural
    success."""
    ix, iy = 2 * eq_idx, 2 * eq_idx + 1
    if payload is None:
        return 0
    history = payload['history']
    if iy >= history.shape[1]:
        return 0
    style = _PIPELINE_STYLES[pipeline]
    lx = np.log10(np.maximum(history[:, ix], 1e-12))
    ly = np.log10(np.maximum(history[:, iy], 1e-12))
    m = np.isfinite(lx) & np.isfinite(ly)
    cloud_label = f'{pipeline} candidates (n={int(m.sum())})'
    ax.scatter(10 ** lx[m], 10 ** ly[m],
               color=style['cloud'], alpha=0.55, s=16, edgecolors='none',
               label=cloud_label)
    drawn = int(m.sum())
    matches_by_eq = payload['matches'] or {}
    eq_matches = matches_by_eq.get(eq_idx)
    if eq_matches is not None and eq_matches.size:
        mx = eq_matches[:, 0]
        my = eq_matches[:, 1]
        mm = np.isfinite(mx) & np.isfinite(my)
        if mm.any():
            ax.scatter(mx[mm], my[mm], color=style['match'],
                       edgecolors='black', linewidths=0.7,
                       marker=style['marker'], s=style['size'], zorder=5,
                       label=f'{pipeline} eq-match (n={int(mm.sum())})')
    return drawn


def _match_count(payload, eq_idx: int = None) -> int:
    """Number of truth-match stars to be plotted.

    With ``eq_idx`` given: count for that specific equation panel
    (used in per-panel titles on coupled systems).
    Without ``eq_idx``: total across all equations (legacy callers).
    """
    if payload is None:
        return 0
    matches_by_eq = payload['matches'] or {}
    if eq_idx is not None:
        arr = matches_by_eq.get(eq_idx)
        if arr is None or arr.size == 0:
            return 0
        return int(np.isfinite(arr).all(axis=1).sum())
    total = 0
    for arr in matches_by_eq.values():
        if arr is None or arr.size == 0:
            continue
        total += int(np.isfinite(arr).all(axis=1).sum())
    return total


def _seed_blurb(payload, label):
    if payload is None:
        return f"{label}: (no history)"
    seed = payload['rec'].get('seed')
    n_cands = payload['history'].shape[0]
    return f"{label}: seed {seed}, {n_cands} unique cands"


def plot_one(legacy, new_, system: str, out_dir: str, show: bool,
             pipelines: tuple = ('legacy', 'new'),
             out_suffix: str = ''):
    """Plot one system with LEGACY (top row) and NEW (bottom row) on
    separate subplots because their objective spaces differ.

    ``legacy`` and ``new_`` are payload dicts from ``_pipeline_payload``
    or ``None`` if that pipeline has no history. ``pipelines`` selects
    which rows to render (e.g. ``('legacy',)`` or ``('new',)`` for the
    split variants). ``out_suffix`` lands inside the output filename
    so the three variants don't overwrite each other.
    """
    payload_by_pipeline = {'legacy': legacy, 'new': new_}
    rows_spec = [(p, payload_by_pipeline[p]) for p in pipelines]
    # If every requested pipeline is missing, skip — nothing to draw.
    if all(p is None for _, p in rows_spec):
        print(f"  [{system}{out_suffix}] no usable history "
              f"for requested pipelines; skipping")
        return None

    n_equations = max(
        (max(1, p['history'].shape[1] // 2)
         for _, p in rows_spec if p is not None),
        default=1,
    )

    # ``sharex='row' / sharey='row'`` ties the equation panels of one
    # pipeline to a single axis range so coupled systems (LV / Lorenz /
    # NS) compare like-for-like across equations. We do NOT share
    # between rows — LEGACY and NEW live in different objective spaces
    # (L2/complexity vs WAPE/instability) and sharing would distort.
    fig, axes = plt.subplots(
        len(rows_spec), n_equations,
        figsize=(5.2 * n_equations, 4.6 * len(rows_spec)),
        squeeze=False,
        sharex='row', sharey='row',
    )

    # Track per-row legend material: (first_drawn_ax, handles, labels)
    # so the legend can be placed centered over the full row of panels
    # (not just above the leftmost column) after tight_layout has
    # positioned the axes.
    row_legend_specs = []
    for row_idx, (pipeline, payload) in enumerate(rows_spec):
        style = _PIPELINE_STYLES[pipeline]
        row_axes = axes[row_idx]
        # Determine this pipeline's per-equation count separately so we
        # only enable panels that exist for the pipeline that's present.
        pipe_n_eq = (max(1, payload['history'].shape[1] // 2)
                     if payload is not None else 0)
        first_drawn_ax = None
        for eq_idx, ax in enumerate(row_axes):
            if payload is None or eq_idx >= pipe_n_eq:
                if payload is None and eq_idx == 0:
                    # Annotate the empty row so the figure shows that
                    # the pipeline has no history available.
                    ax.text(0.5, 0.5,
                            f"{pipeline.upper()}: no sidecar history available",
                            ha='center', va='center',
                            transform=ax.transAxes, fontsize=11,
                            color='dimgray')
                ax.axis('off')
                continue
            _draw_single(ax, payload, eq_idx, pipeline)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(style['x_label'])
            ax.set_ylabel(style['y_label'])
            ax.grid(alpha=0.3, which='both', linewidth=0.4)
            n_match = _match_count(payload, eq_idx)
            if first_drawn_ax is None:
                first_drawn_ax = ax
            # Push every title in the row up by the same pad so the
            # row-spanning legend (which sits just above the axes
            # spines) doesn't overlap titles on the non-legend-host
            # panels. Equation N titles on coupled systems need to
            # line up at the same y across the row.
            ax.set_title(
                f'{pipeline.upper()} — Equation {eq_idx} '
                f'(eq-match n={n_match})',
                fontsize=10, pad=24,
            )
        if first_drawn_ax is not None:
            handles, labels = first_drawn_ax.get_legend_handles_labels()
            row_legend_specs.append((first_drawn_ax, handles, labels))

    cohort_blurbs = []
    if 'new' in pipelines and new_ and new_['matches_from_cohort']:
        cohort_blurbs.append('NEW stars from cohort')
    if 'legacy' in pipelines and legacy and legacy['matches_from_cohort']:
        cohort_blurbs.append('LEGACY stars from cohort')
    cohort_note = f"   ({'; '.join(cohort_blurbs)})" if cohort_blurbs else ''

    # Pick a payload that exists to read epoch metadata.
    ref_payload = next((p for _, p in rows_spec if p is not None), None)
    n_epochs = ref_payload['rec'].get('n_epochs', '?') if ref_payload else '?'
    blurbs = [_seed_blurb(payload_by_pipeline[p], p.upper())
              for p in pipelines]
    fig.suptitle(
        f"{system.upper()}  —  " + '   |   '.join(blurbs)
        + f"   |   {n_epochs} epochs{cohort_note}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.94])

    # Place a figure-level legend per pipeline row, centered over the
    # full width of the figure (not just the leftmost panel) so coupled
    # systems (LV / Lorenz / NS) get a single common legend that spans
    # the equation columns. Anchored to the row's top edge in figure
    # coordinates, read after tight_layout has settled positions.
    for ax_ref, handles, labels in row_legend_specs:
        if not handles:
            continue
        pos = ax_ref.get_position()
        leg_y = pos.y1 + 0.015
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, leg_y),
                   ncol=len(handles), frameon=False, fontsize=8)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'history_match{out_suffix}_{system}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{system}{out_suffix}] wrote {out_path}')
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=_FIG_DIR)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    plotted = []
    for system in SYSTEM_ORDER:
        (legacy_rec, legacy_path), (new_rec, new_path) = _find_paired_with_history(system)
        legacy_payload = _pipeline_payload(system, legacy_rec, legacy_path)
        new_payload = _pipeline_payload(system, new_rec, new_path)
        if legacy_payload is None and new_payload is None:
            print(f'  [skip] {system}: no sidecar history available for '
                  f'either pipeline (re-run with history recording to populate)')
            continue
        if legacy_payload is None:
            print(f'  [{system}] no legacy history; plotting NEW only')
        elif new_payload is None:
            print(f'  [{system}] no new history; plotting LEGACY only')
        # Per-system PNGs split by pipeline plus a joint figure that
        # stacks LEGACY (top) and NEW (bottom) so reviewers can compare
        # at a glance without flipping between files:
        #   history_match_legacy_<system>.png  — LEGACY only.
        #   history_match_new_<system>.png     — NEW only.
        #   history_match_joint_<system>.png   — both pipelines, one row
        #     per pipeline (objective spaces still differ row-to-row).
        out_legacy = plot_one(legacy_payload, new_payload, system,
                              args.out_dir, show=False,
                              pipelines=('legacy',), out_suffix='_legacy')
        out_new = plot_one(legacy_payload, new_payload, system,
                           args.out_dir, show=False,
                           pipelines=('new',), out_suffix='_new')
        out_joint = plot_one(legacy_payload, new_payload, system,
                             args.out_dir, show=False,
                             pipelines=('legacy', 'new'), out_suffix='_joint')
        if out_legacy or out_new or out_joint:
            plotted.append((system, legacy_payload, new_payload))

    # Two grid montages, one per pipeline. Each stays in a single
    # objective space (LEGACY: discrepancy(L2) vs complexity;
    # NEW: discrepancy(WAPE) vs instability).
    for pipeline in ('legacy', 'new'):
        _write_grid_montage(plotted, pipeline, args.out_dir)

    print(f'\nDone. {len(plotted)} per-system figures written.')
    return 0 if plotted else 1




def _write_grid_montage(plotted: list, pipeline: str, out_dir: str):
    """Write a per-pipeline grid montage:
    ``history_match_grid_<pipeline>.png`` — one row per system with
    history for ``pipeline``, columns indexed by equation. All panels
    share the pipeline's objective space (LEGACY: discrepancy(L2) /
    complexity; NEW: discrepancy(WAPE) / instability).
    """
    style = _PIPELINE_STYLES[pipeline]
    # Filter to systems that actually have a payload for this pipeline.
    rows = []
    for system, legacy_payload, new_payload in plotted:
        payload = legacy_payload if pipeline == 'legacy' else new_payload
        if payload is None:
            continue
        rows.append((system, payload))
    if not rows:
        print(f'  [grid:{pipeline}] no systems with history; skipping')
        return None

    ncols = max(1, max(max(1, payload['history'].shape[1] // 2)
                       for _, payload in rows))
    nrows = len(rows)
    # Share scales across the equation columns of a single system row
    # so equations are visually comparable. Different systems live in
    # different objective magnitudes, so don't share across rows.
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.0 * ncols, 3.6 * nrows),
                              squeeze=False,
                              sharex='row', sharey='row')
    first_drawn_ax = None
    for row_idx, (system, payload) in enumerate(rows):
        pipe_n_eq = max(1, payload['history'].shape[1] // 2)
        for eq_idx in range(ncols):
            ax = axes[row_idx][eq_idx]
            if eq_idx >= pipe_n_eq:
                ax.axis('off')
                continue
            _draw_single(ax, payload, eq_idx, pipeline)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(alpha=0.3, which='both', linewidth=0.4)
            if eq_idx == 0:
                ax.set_ylabel(f"{system}\n{style['y_label']}", fontsize=9)
            else:
                ax.set_ylabel(style['y_label'], fontsize=9)
            if row_idx == nrows - 1:
                ax.set_xlabel(style['x_label'], fontsize=9)
            if row_idx == 0:
                # Pad uniformly across the top row so the row-spanning
                # legend sits between the suptitle and the titles
                # without overlapping any of them.
                ax.set_title(f'Equation {eq_idx}', fontsize=10, pad=24)
            if first_drawn_ax is None:
                first_drawn_ax = ax
    handles, labels = ([], [])
    if first_drawn_ax is not None:
        handles, labels = first_drawn_ax.get_legend_handles_labels()
    fig.suptitle(
        f"Candidate-objective history per system — {pipeline.upper()} "
        f"({style['x_label']} vs {style['y_label']})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    if first_drawn_ax is not None and handles:
        pos = first_drawn_ax.get_position()
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, pos.y1 + 0.010),
                   ncol=len(handles), frameon=False, fontsize=9)
    grid_path = os.path.join(out_dir, f'history_match_grid_{pipeline}.png')
    fig.savefig(grid_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {grid_path}')
    return grid_path


if __name__ == '__main__':
    sys.exit(main())
