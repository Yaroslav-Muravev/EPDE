"""
Structural metrics for the thesis Section 4.5 EPDE comparison.

The metric pipeline is text-based: equations are read as strings (the
form produced by EPDE's :meth:`equations(only_str=True)`), parsed into a
canonical token representation that ignores coefficient values and term
ordering, then compared via Hamming distance / equality / modal-set
agreement across repetitions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

# Factor pattern: ``name{key1: val1, key2: val2, ...}`` where ``name`` can
# contain letters, digits, and the symbol characters EPDE uses for
# derivative tokens (``d``, ``u``, ``/``, ``^``, digits) and trig product
# tokens (e.g. ``cos(t)sin(x)``).
_FACTOR_RE = re.compile(r'([A-Za-z0-9_\^/\(\)]+)\s*\{([^}]*)\}')
_PARAM_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,]+)')
_PARAM_ROUND_DIGITS = 3


def _round_param(value: str):
    value = value.strip()
    try:
        return round(float(value), _PARAM_ROUND_DIGITS)
    except ValueError:
        return value


def _parse_factor(text: str):
    """Return ``(name, frozenset_of_param_items)`` or None if no factor."""
    m = _FACTOR_RE.search(text)
    if m is None:
        return None
    name = m.group(1)
    params_str = m.group(2)
    params = {}
    for pm in _PARAM_RE.finditer(params_str):
        params[pm.group(1)] = _round_param(pm.group(2))
    return (name, frozenset(params.items()))


def _parse_term(term_text: str):
    """Parse a single ``c * f1{...} * f2{...}`` term into a frozenset of factors.

    Pure-constant terms (e.g. ``0.0``) and terms whose leading coefficient
    is numerically zero are filtered out by returning None.
    """
    pieces = [p.strip() for p in term_text.split('*')]
    factors = []
    coef = 1.0
    coef_seen = False
    for piece in pieces:
        if not piece:
            continue
        factor = _parse_factor(piece)
        if factor is None:
            # piece is a bare numeric coefficient (or unparseable scalar).
            try:
                val = float(piece)
                coef *= val
                coef_seen = True
                continue
            except ValueError:
                # Unrecognised piece: skip rather than crash; the canonical
                # set will simply omit it (and Hamming will reflect that).
                continue
        factors.append(factor)

    if not factors:
        # Pure-constant or unparseable term -> drop.
        return None
    if coef_seen and abs(coef) < 1e-12:
        # Zero coefficient -> term doesn't actually appear in the equation.
        return None
    return frozenset(factors)


def _canonical_equation(eq_text: str):
    """Parse one equation ``rhs_sum = target`` into a canonical tuple.

    Returns ``(target_term, frozenset_of_rhs_terms)`` or None if no ``=``.
    """
    if '=' not in eq_text:
        return None
    left, right = eq_text.split('=', 1)
    target_term = _parse_term(right)
    rhs_terms = []
    for term_text in left.split('+'):
        term = _parse_term(term_text)
        if term is not None:
            rhs_terms.append(term)
    return (target_term, frozenset(rhs_terms))


def canonical_tokens(eq_texts: Sequence[str]) -> frozenset:
    """Convert a list of equation text strings into a canonical structure.

    Each equation contributes one element to the returned frozenset:
    ``(target_term, frozenset_of_rhs_terms)``. The result ignores
    coefficient magnitudes, term ordering, and factor ordering within
    terms; it preserves factor names + parameters (powers, freqs, dims)
    rounded to :data:`_PARAM_ROUND_DIGITS` digits.
    """
    out = []
    for eq in eq_texts:
        if not eq.strip():
            continue
        canon = _canonical_equation(eq)
        if canon is not None:
            out.append(canon)
    return frozenset(out)


def hamming(discovered: frozenset, truth: frozenset) -> int:
    """Term-level structural distance between two canonical equation systems.

    Equations are matched by their target (LHS) term. For each matched
    target, the contribution is the cardinality of the symmetric
    difference of the right-hand-side term sets — so a single missing or
    extra rhs term costs 1. For equations whose target exists in only
    one side, the cost is ``1 + len(rhs)`` (target mismatch plus all its
    rhs terms). A pure-constant (`+ 0.0`) term is filtered out at
    canonicalisation time and never contributes.

    Examples (Lorenz first equation only):
        truth = {(du/dt, {a, b, c})}, discovered = {(du/dt, {a, b})}
            -> hamming = 1   (one rhs term missing)
        truth = {(du/dt, {a, b})},    discovered = {(dv/dt, {a, b})}
            -> hamming = 1 + 2 + 1 + 2 = 6  (target differs, both sides counted)
    """
    truth_by_target = {target: rhs for target, rhs in truth}
    disc_by_target = {target: rhs for target, rhs in discovered}

    total = 0
    for target in set(truth_by_target) | set(disc_by_target):
        truth_rhs = truth_by_target.get(target)
        disc_rhs = disc_by_target.get(target)
        if truth_rhs is None:
            total += 1 + len(disc_rhs)
        elif disc_rhs is None:
            total += 1 + len(truth_rhs)
        else:
            total += len(truth_rhs.symmetric_difference(disc_rhs))
    return total


def structural_success(discovered: frozenset, truth: frozenset) -> bool:
    """True iff ``discovered`` equals ``truth`` as a canonical system."""
    return hamming(discovered, truth) == 0


def consistency_rate(reps_canonical: Iterable[frozenset]) -> float:
    """Fraction of reps whose canonical system equals the modal canonical system."""
    reps = list(reps_canonical)
    if not reps:
        return 0.0
    counts = Counter(reps)
    modal_count = counts.most_common(1)[0][1]
    return modal_count / len(reps)


def wilson_ci(successes: int, n: int, z: float = 1.96):
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


if __name__ == '__main__':
    # Quick self-check: round-trip the Lorenz triple and confirm Hamming == 0
    # against itself, then perturb one term and confirm Hamming == 2.
    lorenz_truth = [
        '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} = du/dx0{power: 1.0}',
        '28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} = dv/dx0{power: 1.0}',
        '1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} = dw/dx0{power: 1.0}',
    ]
    canon_truth = canonical_tokens(lorenz_truth)
    print('canon_truth size:', len(canon_truth))
    assert hamming(canon_truth, canon_truth) == 0
    assert structural_success(canon_truth, canon_truth)

    perturbed = list(lorenz_truth)
    # Drop the -10*u term from the first equation -> one rhs term missing.
    perturbed[0] = '10.0 * v{power: 1.0} = du/dx0{power: 1.0}'
    canon_perturbed = canonical_tokens(perturbed)
    h = hamming(canon_perturbed, canon_truth)
    print('hamming(1 term missing) =', h)
    assert h == 1, f"expected 1, got {h}"

    # Swap one rhs term for a different one: 1 removed + 1 added = 2.
    perturbed2 = list(lorenz_truth)
    perturbed2[0] = ('10.0 * v{power: 1.0} + -10.0 * u{power: 2.0} '
                     '= du/dx0{power: 1.0}')
    h2 = hamming(canonical_tokens(perturbed2), canon_truth)
    print('hamming(1 term swapped) =', h2)
    assert h2 == 2, f"expected 2, got {h2}"

    # Adding a pure-constant `+ 0.0` term must NOT change the canonical form.
    with_zero = list(lorenz_truth)
    with_zero[0] = '10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    h_zero = hamming(canonical_tokens(with_zero), canon_truth)
    print('hamming(+0.0 added) =', h_zero)
    assert h_zero == 0, f"expected 0, got {h_zero}"

    # Drop a whole equation -> target + its 2 rhs terms = 3.
    perturbed3 = list(lorenz_truth[:2])
    h3 = hamming(canonical_tokens(perturbed3), canon_truth)
    print('hamming(1 equation missing) =', h3)
    assert h3 == 3, f"expected 3, got {h3}"

    print('thesis_metrics self-check OK')
