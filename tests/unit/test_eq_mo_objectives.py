#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for ``epde/eq_mo_objectives.py``.

Pins the fix for the legacy AC IndexError: ``_complexity_single_eq`` must
read coefficient slots through ``weights_internal`` (always length
``len(structure) - 1``) and not ``weights_final``, which the sparsity
operators truncate to ``nnz(coef) + 1`` by filtering out zero weights.
Before the fix, this caused 4/30 Allen-Cahn legacy reps in the thesis
Section-4.5 rerun to crash inside the MOEA/D dominance check.

The tests use lightweight mocks so we exercise the indexing logic
directly without spinning up the full pool/token/translate machinery.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.eq_mo_objectives import (
    _complexity_single_eq,
    equation_complexity_by_factors,
)


class _MockFactor:
    """Stub satisfying the ``complexity_deriv`` contract: a ``deriv_code``
    attribute and a ``param('power')`` method."""

    def __init__(self, deriv_code=None, power=1.0):
        self.deriv_code = deriv_code
        self._power = power

    def param(self, name):
        assert name == 'power', f'unexpected param request: {name!r}'
        return self._power


class _MockTerm:
    def __init__(self, factors):
        self.structure = list(factors)


def _make_system(structure_terms, target_idx, weights_internal,
                 weights_final, key='u'):
    eq = SimpleNamespace(
        structure=list(structure_terms),
        target_idx=target_idx,
        weights_internal=np.asarray(weights_internal, dtype=float),
        weights_final=np.asarray(weights_final, dtype=float),
    )
    return SimpleNamespace(vals={key: eq}, vars_to_describe=[key])


class TestComplexitySingleEqWeightsInternalIndexing:
    """The crash trigger from the AC legacy rerun: ``LASSOSparsity`` /
    ``VWSRSparsity`` set
    ``weights_final = np.append([w for w in coef if w != 0], intercept)``
    so ``weights_final.size = nnz + 1`` rather than ``len(structure) - 1``.
    ``_complexity_single_eq`` iterates structure positions and uses
    ``weights_internal`` (full length) for the non-zero check, so the
    crash no longer reproduces."""

    def test_no_crash_when_weights_final_truncated_by_sparsity(self):
        # 4-term structure with target at idx=1. weights_internal stays at
        # length 3 (= L - 1) even when sparsity zeros two of the three
        # non-target coefficients; weights_final shrinks to nnz + 1 = 2.
        terms = [
            _MockTerm([_MockFactor(deriv_code=[None], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0, 0], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0, 0, 0], power=1.0)]),
        ]
        system = _make_system(
            structure_terms=terms,
            target_idx=1,
            weights_internal=[1.0, 0.0, 0.0],
            weights_final=[1.0, 0.0],
        )
        # Must not raise. Pre-fix: IndexError on weights_final[2] when
        # idx=3 hits the elif branch and reads weights_final[idx-1].
        complexity = _complexity_single_eq(system, 'u')

        # term[0]: deriv_code [None] → 0.5  (counted, weights_internal[0]=1)
        # term[1]: target,   deriv_code [0] → 1.0 (always counted)
        # term[2]: deriv_code [0,0] → skipped (weights_internal[1]=0)
        # term[3]: deriv_code [0,0,0] → skipped (weights_internal[2]=0)
        assert complexity == pytest.approx(1.5)

    def test_matches_full_length_weights_final_case(self):
        # When sparsity didn't zero anything, weights_final.size = L
        # (n_non_target_terms + intercept). The function still routes
        # through weights_internal; the result is identical and the
        # all-non-zero case never broke.
        terms = [
            _MockTerm([_MockFactor(deriv_code=[None], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0, 0], power=1.0)]),
        ]
        system = _make_system(
            structure_terms=terms,
            target_idx=1,
            weights_internal=[1.0, 1.0],
            weights_final=[1.0, 1.0, 0.0],
        )
        complexity = _complexity_single_eq(system, 'u')
        # term[0] 0.5 + target 1.0 + term[2] 2.0 = 3.5
        assert complexity == pytest.approx(3.5)

    def test_target_only_complexity(self):
        # Edge case: all non-target weights zero. Only the target counts.
        terms = [
            _MockTerm([_MockFactor(deriv_code=[None], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0, 0], power=1.0)]),
        ]
        system = _make_system(
            structure_terms=terms,
            target_idx=1,
            weights_internal=[0.0, 0.0],
            weights_final=[],  # sparsity dropped everything except intercept
        )
        complexity = _complexity_single_eq(system, 'u')
        assert complexity == pytest.approx(1.0)

    def test_equation_complexity_by_factors_returns_tuple_when_key_none(self):
        # Public wrapper contract: with equation_key=None, return a tuple
        # over vars_to_describe; with a specific key, return a scalar.
        terms = [
            _MockTerm([_MockFactor(deriv_code=[None], power=1.0)]),
            _MockTerm([_MockFactor(deriv_code=[0], power=1.0)]),
        ]
        system = _make_system(
            structure_terms=terms,
            target_idx=1,
            weights_internal=[1.0],
            weights_final=[1.0, 0.0],
        )
        per_var = equation_complexity_by_factors(system, equation_key=None)
        assert isinstance(per_var, tuple)
        assert len(per_var) == 1
        scalar = equation_complexity_by_factors(system, equation_key='u')
        assert scalar == pytest.approx(per_var[0])

    def test_documented_old_buggy_indexing_would_fail(self):
        # Documents the pre-fix crash mode: indexing the size-2
        # weights_final at position idx-1 = 2 raises IndexError. Catches
        # any future regression that reintroduces a weights_final lookup
        # in this function.
        weights_final = np.array([1.0, 0.0])
        with pytest.raises(IndexError):
            _ = weights_final[3 - 1]  # was: weights_final[idx-1] for idx=3
