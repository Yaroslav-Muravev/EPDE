"""
Shared runner module for the thesis Section 4.5 within-platform comparison.

This module is the single source of truth for:
    * the 8-cell pipeline table (``_PIPELINE_SETTINGS`` -> ``pipeline_settings``)
    * the ``SystemCfg`` dataclass consumed by ``build_search`` / ``run_one``
    * ``run_smoke`` (batched per-rep JSON dumps with resume-on-restart)
    * ``load_config`` for parsing per-system YAML configs

Per-system configuration lives at:

    projects/thesis/configs/<name>.yaml       (declarative: truth equations,
                                               output dir, data_fun_pow, ...)
    projects/thesis/adapters/<name>.py        (Python: load_data(),
                                               build_extra_tokens(coords, dim))

Pipeline selection (``legacy`` vs ``new`` is the main thesis comparison; the
six ablation cells off the 000/111 diagonal cover the 2x2x2 factorial):

    LEGACY  -> L2Fitness   + LASSOSparsity + use_pic=False
    NEW     -> L2LRFitness + VWSRSparsity  + use_pic=True
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch

# Make sure the EPDE package is importable when running this module's CLI
# entries directly (``python projects/thesis/run.py lv``).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from epde.interface.interface import EpdeSearch  # noqa: E402
from epde.operators.common.fitness import L2Fitness, L2LRFitness  # noqa: E402
from epde.operators.common.sparsity import LASSOSparsity, VWSRSparsity  # noqa: E402
from epde import GridTokens  # noqa: E402


CONFIGS_DIR = os.path.join(_THIS_DIR, 'configs')
ADAPTERS_DIR = os.path.join(_THIS_DIR, 'adapters')
RESULTS_DIR = os.path.join(_THIS_DIR, 'results')


# Full 2x2x2 ablation table for the three thesis-NEW contributions:
# (1) WAPE fitness      -> L2LRFitness   vs LEGACY L2Fitness
# (2) Instability obj   -> use_pic=True swaps MOEA/D's 2nd objective
#                          (equation_terms_stability) vs LEGACY
#                          (equation_complexity_by_factors)
# (3) Novel regularizer -> VWSRSparsity (PhysicsInformedLasso, CV-weighted)
#                          vs LEGACY LASSOSparsity (sklearn.Lasso)
_PIPELINE_SETTINGS = {
    'legacy':      {'fitness_cls': L2Fitness,   'sparsity_cls': LASSOSparsity, 'use_pic': False},
    'wape':        {'fitness_cls': L2LRFitness, 'sparsity_cls': LASSOSparsity, 'use_pic': False},
    'instab':      {'fitness_cls': L2Fitness,   'sparsity_cls': LASSOSparsity, 'use_pic': True},
    'reg':         {'fitness_cls': L2Fitness,   'sparsity_cls': VWSRSparsity,  'use_pic': False},
    'wape_instab': {'fitness_cls': L2LRFitness, 'sparsity_cls': LASSOSparsity, 'use_pic': True},
    'wape_reg':    {'fitness_cls': L2LRFitness, 'sparsity_cls': VWSRSparsity,  'use_pic': False},
    'instab_reg':  {'fitness_cls': L2Fitness,   'sparsity_cls': VWSRSparsity,  'use_pic': True},
    'new':         {'fitness_cls': L2LRFitness, 'sparsity_cls': VWSRSparsity,  'use_pic': True},
}

# Default pipelines for the main Section 4.5 comparison.
PIPELINES = ('legacy', 'new')

# Off-diagonal cells of the 2x2x2 factorial -- pass to ``run_smoke`` as the
# ``pipelines`` argument from the ablation entry point. Excludes
# ``legacy`` and ``new`` since their reps live in the default results tree
# (the 000 and 111 corners of the cube).
ABLATION_PIPELINES = (
    'wape', 'instab', 'reg',
    'wape_instab', 'wape_reg', 'instab_reg',
)


def pipeline_settings(pipeline: str) -> dict:
    """Return ``EpdeSearch`` kwargs for a single pipeline label.

    Recognises the original two labels (``legacy``, ``new``) and the six
    off-diagonal ablation labels. Forward the returned dict directly to
    :class:`EpdeSearch` (``use_pic``, ``fitness_cls``, ``sparsity_cls``).
    """
    try:
        return dict(_PIPELINE_SETTINGS[pipeline])
    except KeyError:
        raise ValueError(
            f"Unknown pipeline {pipeline!r}; expected one of {tuple(_PIPELINE_SETTINGS)}"
        )


@dataclass
class SystemCfg:
    """Per-system configuration consumed by :func:`run_one`.

    name: short system identifier used in output filenames.
    truth_tokens: canonical token set encoding the ground-truth equations.
    outdir: directory to write per-rep JSON results into.
    load_data: callable returning ``(coordinate_tensors, data_list,
        variable_names, dimensionality)``. ``dimensionality`` is ``0`` for
        ODE systems and the number of spatial axes for PDE systems.
    build_extra_tokens: optional callable returning extra EPDE tokens
        beyond the auto-added ``GridTokens``. Signature
        ``(coords, dim) -> list``. Default: returns ``[]``.
    """

    name: str
    truth_tokens: frozenset
    outdir: str
    load_data: Callable[[], tuple]
    build_extra_tokens: Callable[[Any, int], list] = field(
        default_factory=lambda: (lambda coords, dim: [])
    )
    data_fun_pow: int = 3
    early_stop_on_truth: bool = True


def load_config(name_or_path: str) -> SystemCfg:
    """Resolve a YAML config into a fully-populated :class:`SystemCfg`.

    ``name_or_path`` may be a bare system name (e.g. ``"lv"`` -- looked up
    as ``configs/lv.yaml``) or an explicit path to a YAML file (relative
    paths are resolved against the current working directory).

    Schema (see ``configs/<name>.yaml`` for examples):
        name: str                                 # required
        truth_equations: list[str]                # required; canonicalised at load time
        adapter: str                              # optional; defaults to ``name``
        outdir: str                               # optional; defaults to ``name`` (under results/)
        data_fun_pow: int                         # optional; default 3
        early_stop_on_truth: bool                 # optional; default True

    Returns the populated dataclass. The adapter module is imported via
    ``importlib`` from ``projects/thesis/adapters/<adapter>.py``; it must
    export ``load_data`` and may optionally export ``build_extra_tokens``.
    """
    import yaml  # local import: only required when YAML configs are used

    yaml_path = (
        name_or_path
        if os.path.sep in name_or_path or name_or_path.endswith('.yaml')
        else os.path.join(CONFIGS_DIR, f'{name_or_path}.yaml')
    )
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"config not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        d = yaml.safe_load(f) or {}

    name = d.get('name')
    if not name:
        raise ValueError(f"{yaml_path}: 'name' is required")

    from thesis_metrics import canonical_tokens
    truth_equations = d.get('truth_equations') or []
    truth_tokens = canonical_tokens(truth_equations)

    adapter_name = d.get('adapter', name)
    if ADAPTERS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    adapter_mod = importlib.import_module(f'adapters.{adapter_name}')

    if not hasattr(adapter_mod, 'load_data'):
        raise AttributeError(
            f"adapter {adapter_name!r} must export load_data() -> "
            "(coords, data, variable_names, dim)"
        )

    outdir_rel = d.get('outdir', name)
    outdir = (
        outdir_rel
        if os.path.isabs(outdir_rel)
        else os.path.abspath(os.path.join(RESULTS_DIR, outdir_rel))
    )

    kwargs: dict = dict(
        name=name,
        truth_tokens=truth_tokens,
        outdir=outdir,
        load_data=adapter_mod.load_data,
    )
    if hasattr(adapter_mod, 'build_extra_tokens'):
        kwargs['build_extra_tokens'] = adapter_mod.build_extra_tokens
    if 'data_fun_pow' in d:
        kwargs['data_fun_pow'] = int(d['data_fun_pow'])
    if 'early_stop_on_truth' in d:
        kwargs['early_stop_on_truth'] = bool(d['early_stop_on_truth'])
    return SystemCfg(**kwargs)


def _boundary_for(coords) -> Any:
    """Return ``10%``-of-axis boundary for the supplied EPDE coordinate tensors.

    ODE problems pass a single 1-D array via ``(t,)``; the returned
    boundary is a scalar ``len(t) // 10``. PDE problems pass a meshgrid
    tuple where every array has the same multidimensional shape; the
    returned boundary is a per-axis tuple of ``axis_size // 10``.
    """
    sample = np.asarray(coords[0])
    if sample.ndim <= 1:
        return max(1, len(sample) // 10)
    return tuple(max(1, n // 10) for n in sample.shape)


def _build_truth_match_callback(cfg: 'SystemCfg') -> Callable:
    """Return a per-epoch callback that stops MOEA/D once any Pareto-0
    candidate canonically matches ``cfg.truth_tokens``.
    """
    from thesis_metrics import canonical_tokens, structural_success
    truth = cfg.truth_tokens

    def _cb(snapshot, epoch_idx):
        for entry in snapshot:
            text = entry.get('text_form', '') if isinstance(entry, dict) else str(entry)
            lines = [line for line in text.split('\n') if line.strip()]
            try:
                canon = canonical_tokens(lines)
            except Exception:
                continue
            if structural_success(canon, truth):
                return True
        return False

    return _cb


def build_search(cfg: 'SystemCfg', pipeline_kwargs: dict) -> EpdeSearch:
    """Universal EPDE search builder for the thesis Section 4.5 comparison.

    Hyperparameters are uniform across all benchmark systems; the only
    branches are ODE vs PDE (deriv order, grid-token labels, boundary
    shape) and the per-system data / extra-token callbacks declared on
    ``cfg``. Pipeline selection is forwarded through ``pipeline_kwargs``.
    """
    coords, data, variable_names, dim = cfg.load_data()
    boundary = _boundary_for(coords)
    max_deriv_order = (2,) if dim == 0 else (2, 4)

    grid_labels = ['x_0'] if dim == 0 else [f'x_{i}' for i in range(dim + 1)]
    grid_tokens = GridTokens(grid_labels, dimensionality=dim, max_power=2)
    additional_tokens = [grid_tokens] + list(cfg.build_extra_tokens(coords, dim))

    search = EpdeSearch(
        use_solver=False,
        multiobjective_mode=True,
        boundary=boundary,
        coordinate_tensors=coords,
        verbose_params={'show_iter_idx': True},
        device='cuda',
        **pipeline_kwargs,
    )
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})

    early_stop_cb = _build_truth_match_callback(cfg) if cfg.early_stop_on_truth else None
    search.set_moeadd_params(population_size=16, training_epochs=5,
                             early_stopping_callback=early_stop_cb)

    search.fit(
        data=data,
        variable_names=variable_names,
        max_deriv_order=max_deriv_order,
        derivs=None,
        equation_terms_max_number=10,
        data_fun_pow=cfg.data_fun_pow,
        deriv_fun_pow=2,
        additional_tokens=additional_tokens,
        equation_factors_max_number={'factors_num': [1, 2], 'probas': [0.65, 0.35]},
        eq_sparsity_interval=(1e-5, 1e0),
        fourier_layers=False,
    )
    return search


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokens_to_json(tokens) -> list:
    """Recursively convert a canonical token structure into JSON-friendly lists."""
    def factor(f):
        name, params = f
        return [name, sorted(([k, v] for k, v in params), key=lambda p: p[0])]

    def term(t):
        return sorted([factor(f) for f in t], key=lambda f: (f[0], repr(f[1])))

    out = []
    for target, rhs in tokens:
        out.append([
            term(target),
            sorted((term(t) for t in rhs), key=lambda x: repr(x)),
        ])
    return sorted(out, key=lambda x: repr(x))


def _discovery_epochs(final_token_sets, pareto_history) -> list:
    """For each canonical token set in ``final_token_sets``, return the
    first epoch index (0-based) in ``pareto_history`` whose Pareto-0
    snapshot contains a solution with the same canonical structure.
    """
    from thesis_metrics import canonical_tokens

    snapshot_canon = []
    for epoch_snapshot in pareto_history:
        per_epoch = []
        for sol_record in epoch_snapshot:
            text = sol_record.get('text_form', '') if isinstance(sol_record, dict) else str(sol_record)
            lines = [line for line in text.split('\n') if line.strip()]
            per_epoch.append(canonical_tokens(lines))
        snapshot_canon.append(per_epoch)

    epochs = []
    for target in final_token_sets:
        first = None
        for epoch_idx, epoch_canon in enumerate(snapshot_canon):
            if any(c == target for c in epoch_canon):
                first = epoch_idx
                break
        epochs.append(first)
    return epochs


def _extract_discovered(search: EpdeSearch) -> list:
    """Return all solutions from the non-dominated Pareto level."""
    eqs = search.equations(only_print=False, only_str=True, num=1)
    if not eqs:
        return []
    if isinstance(eqs[0], list):
        level0_solutions = eqs[0]
    else:
        level0_solutions = eqs

    out = []
    for solution in level0_solutions:
        if not isinstance(solution, str):
            solution = str(solution)
        out.append([line for line in solution.split('\n') if line.strip()])
    return out


def _extract_objectives(search: EpdeSearch) -> list:
    """Return per-solution objective vectors aligned with ``_extract_discovered``."""
    try:
        level0 = search.optimizer.pareto_levels.levels[0]
    except Exception:
        return []
    out = []
    for sol in level0:
        try:
            obj = sol.obj_fun.tolist() if hasattr(sol.obj_fun, 'tolist') else list(sol.obj_fun)
        except Exception:
            obj = None
        out.append(obj)
    return out


def run_one(system_cfg: SystemCfg, pipeline: str, seed: int) -> dict:
    """Run a single (system, pipeline, seed) repetition.

    Returns a dict suitable for JSON serialization. Exceptions are caught
    and recorded as ``error`` and ``traceback`` fields so a failing rep
    does not kill the batch.
    """
    from thesis_metrics import canonical_tokens, hamming, structural_success

    pipeline_kwargs = pipeline_settings(pipeline)
    _set_seeds(seed)

    record: dict = {
        'system': system_cfg.name,
        'pipeline': pipeline,
        'seed': seed,
        'pipeline_kwargs': {
            'use_pic': pipeline_kwargs['use_pic'],
            'fitness_cls': pipeline_kwargs['fitness_cls'].__name__,
            'sparsity_cls': pipeline_kwargs['sparsity_cls'].__name__,
        },
    }

    t0 = time.time()
    try:
        search = build_search(system_cfg, pipeline_kwargs)
        elapsed = time.time() - t0
        solutions_text = _extract_discovered(search)
        objectives_per_solution = _extract_objectives(search)
        per_solution_tokens = [canonical_tokens(sol) for sol in solutions_text]
        pareto_history = list(getattr(search, 'pareto_history', []))
        if per_solution_tokens:
            hammings = [hamming(c, system_cfg.truth_tokens) for c in per_solution_tokens]
            best_idx = int(min(range(len(hammings)), key=lambda i: hammings[i]))
            discovery_epochs = _discovery_epochs(per_solution_tokens, pareto_history)
            best_objectives = (
                objectives_per_solution[best_idx]
                if best_idx < len(objectives_per_solution) else None
            )
            record.update({
                'runtime_sec': elapsed,
                'n_pareto_solutions': len(per_solution_tokens),
                'discovered_text_per_solution': solutions_text,
                'discovered_text': solutions_text[best_idx],
                'discovered_tokens_per_solution': [_tokens_to_json(c) for c in per_solution_tokens],
                'discovered_tokens': _tokens_to_json(per_solution_tokens[best_idx]),
                'truth_tokens': _tokens_to_json(system_cfg.truth_tokens),
                'hamming_per_solution': hammings,
                'hamming': hammings[best_idx],
                'discovery_epoch_per_solution': discovery_epochs,
                'discovery_epoch': discovery_epochs[best_idx],
                'n_epochs': len(pareto_history),
                'objectives_per_solution': objectives_per_solution,
                'objectives': best_objectives,
                'structural_success': any(
                    structural_success(c, system_cfg.truth_tokens) for c in per_solution_tokens
                ),
            })
        else:
            record.update({
                'runtime_sec': elapsed,
                'n_pareto_solutions': 0,
                'discovered_text_per_solution': [],
                'discovered_text': [],
                'discovered_tokens_per_solution': [],
                'discovered_tokens': [],
                'truth_tokens': _tokens_to_json(system_cfg.truth_tokens),
                'hamming_per_solution': [],
                'hamming': None,
                'objectives_per_solution': [],
                'objectives': None,
                'structural_success': False,
            })
    except Exception as exc:  # pragma: no cover - smoke-time diagnostic
        record.update({
            'runtime_sec': time.time() - t0,
            'error': repr(exc),
            'traceback': traceback.format_exc(),
            'n_pareto_solutions': 0,
            'discovered_text_per_solution': [],
            'discovered_text': [],
            'discovered_tokens': [],
            'hamming': None,
            'objectives_per_solution': [],
            'objectives': None,
            'structural_success': False,
        })
    return record


def _resolve_out_root(system_cfg: SystemCfg, outdir: Optional[str]) -> str:
    """Resolve the final output directory for a batch.

    Default (``outdir is None``)    -> ``system_cfg.outdir`` (typically
                                       ``projects/thesis/results/<name>``).
    Absolute path                   -> used as-is.
    Bare tag (e.g. ``ablation_v2``) -> ``results/<tag>/<name>``, so a
                                       tagged sweep across all systems
                                       stays grouped under one folder
                                       (``results/<tag>/lv``, .../lorenz, ...)
                                       and the aggregator can scan a tag in
                                       one glob.
    """
    if outdir is None:
        return system_cfg.outdir
    if os.path.isabs(outdir):
        return outdir
    return os.path.join(RESULTS_DIR, outdir, system_cfg.name)


def run_smoke(
    system_cfg: SystemCfg,
    reps: int = 3,
    pipelines: Iterable[str] = PIPELINES,
    seed_base: int = 0,
    resume: bool = True,
    outdir: Optional[str] = None,
) -> None:
    """Run ``reps`` × len(pipelines) repetitions and write JSON per rep.

    With ``resume=True`` (default) any ``(pipeline, rep)`` whose target JSON
    already exists and parses as JSON is skipped. Pass ``resume=False`` to
    overwrite. See :func:`_resolve_out_root` for ``outdir`` semantics.
    """
    out_root = _resolve_out_root(system_cfg, outdir)
    os.makedirs(out_root, exist_ok=True)
    for pipeline in pipelines:
        for rep in range(reps):
            seed = seed_base + rep
            out_path = os.path.join(out_root, f"{pipeline}_rep{rep:02d}.json")
            if resume and os.path.exists(out_path):
                try:
                    with open(out_path, 'r', encoding='utf-8') as fh:
                        json.load(fh)
                    print(f"\n========== {system_cfg.name} / {pipeline} / rep {rep} -- "
                          f"skipping (resume; {out_path} exists) ==========")
                    continue
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"[resume] {out_path} unreadable ({exc!r}); re-running rep")
            print(f"\n========== {system_cfg.name} / {pipeline} / rep {rep} (seed={seed}) ==========")
            record = run_one(system_cfg, pipeline, seed)
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(record, fh, indent=2, default=str)
            status = 'OK' if 'error' not in record else 'FAIL'
            ham = record.get('hamming')
            epoch = record.get('discovery_epoch')
            n_ep = record.get('n_epochs')
            epoch_str = f"epoch={epoch}/{n_ep}" if epoch is not None else "epoch=?"
            print(f"  -> {status}  hamming={ham}  {epoch_str}  time={record.get('runtime_sec', 0.0):.1f}s")
            print(f"  -> saved {out_path}")
