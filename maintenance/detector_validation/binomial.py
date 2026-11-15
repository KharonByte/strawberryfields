"""Exact-binomial inversion with outward Decimal recurrence enclosures.

This is numerical enclosure of the exact binomial law, not a normal approximation.
No SciPy, beta-function package or simulator is used.
"""

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache


@lru_cache(maxsize=4096)
def binomial_cdf(n, x, q):
    """Enclose Pr(Binomial(n,q)<=x), for an exact rational q in [0,1]."""
    q = F(q)
    if not 1 <= n <= 100000 or not 0 <= q <= 1:
        raise ValueError("binomial parameters exceed fixed design")
    if x < 0:
        return F(0), F(0)
    if x >= n or q == 0:
        return F(1), F(1)
    if q == 1:
        return F(0), F(0)
    # Complement so recurrence length is at most half of n.
    if x > n - x - 1:
        lo, hi = binomial_cdf(n, n - x - 1, 1 - q)
        return 1 - hi, 1 - lo
    answers = []
    for rounding in (ROUND_FLOOR, ROUND_CEILING):
        with localcontext() as ctx:
            ctx.prec = 70
            ctx.rounding = rounding
            # Bisection uses dyadic q; these conversions are exact within 70 digits.
            d = Decimal(q.numerator) / Decimal(q.denominator)
            if F(d) != q:
                raise ValueError("CDF point not exactly represented in Decimal")
            one_minus = Decimal(1) - d
            # Repeated integer multiplication with directed rounding, no Decimal pow/log.
            term, base, exponent = Decimal(1), one_minus, n
            while exponent:
                if exponent & 1:
                    term *= base
                exponent >>= 1
                if exponent:
                    base *= base
            total = term
            for k in range(x):
                term = term * Decimal(n - k) / Decimal(k + 1) * d / one_minus
                total += term
            answers.append(F(total))
    return max(F(0), answers[0]), min(F(1), answers[1])


def _root(n, x, target, width):
    lo, hi = F(0), F(1)
    while hi - lo > width:
        mid = (lo + hi) / 2
        cdf_lo, cdf_hi = binomial_cdf(n, x, mid)
        # CDF is decreasing in q; retain an enclosing root bracket.
        if cdf_lo > target:
            lo = mid
        elif cdf_hi < target:
            hi = mid
        else:
            raise ValueError("binomial arithmetic cannot resolve monotone bisection")
    return lo, hi


@lru_cache(maxsize=128)
def clopper_pearson(n, x, a=F(3, 68000), width=F(1, 10**10)):
    """Return outward brackets for both equal-tailed CP endpoints."""
    if type(x) is not int or not 0 <= x <= n or not 0 < a < F(1, 2):
        raise ValueError("invalid binomial interval request")
    if not F(1, 10**12) <= width <= F(1, 10**10):
        raise ValueError("endpoint width outside reviewed design")
    lower = (F(0), F(0)) if x == 0 else _root(n, x - 1, 1 - a, width)
    upper = (F(1), F(1)) if x == n else _root(n, x, a, width)
    return {
        "lower_bracket": lower,
        "upper_bracket": upper,
        "lower_outward": lower[0],
        "upper_outward": upper[1],
        "endpoint_width_max": max(lower[1] - lower[0], upper[1] - upper[0]),
        "method": "exact-binomial recurrence with directed70-digit arithmetic and dyadic bisection",
    }
