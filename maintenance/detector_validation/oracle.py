"""Independent analytic detector laws with rational interval certificates.

No simulator, numerical package, random source, or candidate probability helper is used.
Inputs denote mathematical decimal parameters. Optional binary envelopes include each
literal's binary64 representation; they do not certify a candidate's state preparation.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import json
import math

PRECISION = 80
SCALE = 10**PRECISION
EPS = F(1, SCALE)


@dataclass(frozen=True)
class Interval:
    """Closed interval with exact rational endpoints."""

    lo: F
    hi: F

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("reversed interval")
        # Outward fixed-grid rounding keeps exact rational endpoints bounded in size.
        # Each operation therefore encloses its exact result without denominator blowup.
        object.__setattr__(self, "lo", F(self.lo.numerator * SCALE // self.lo.denominator, SCALE))
        object.__setattr__(
            self, "hi", F(-((-self.hi.numerator * SCALE) // self.hi.denominator), SCALE)
        )

    @classmethod
    def point(cls, x):
        q = x if isinstance(x, F) else F(str(x))
        return cls(q, q)

    def __add__(self, other):
        other = interval(other)
        return Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        other = interval(other)
        ends = [x * y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Interval(min(ends), max(ends))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = interval(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("division interval contains zero")
        return self * Interval(1 / other.hi, 1 / other.lo)

    def __rtruediv__(self, other):
        return interval(other) / self

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("only nonnegative integer powers")
        if n == 0:
            return interval(1)
        if n % 2 == 0 and self.lo <= 0 <= self.hi:
            return Interval(F(0), max(abs(self.lo), abs(self.hi)) ** n)
        ends = [self.lo**n, self.hi**n]
        return Interval(min(ends), max(ends))

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("negative square root interval")
        low = math.isqrt(self.lo.numerator * SCALE**2 // self.lo.denominator)
        high = math.isqrt(self.hi.numerator * SCALE**2 // self.hi.denominator)
        exact_high = F(high, SCALE) ** 2 == self.hi
        return Interval(F(low, SCALE), F(high + (not exact_high), SCALE))

    def exp(self):
        left, right = _exp_point(self.lo), _exp_point(self.hi)
        return Interval(left.lo, right.hi)

    def outward_strings(self):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            ctx.rounding = ROUND_FLOOR
            lo = str(Decimal(self.lo.numerator) / Decimal(self.lo.denominator))
            ctx.rounding = ROUND_CEILING
            hi = str(Decimal(self.hi.numerator) / Decimal(self.hi.denominator))
        return {"lower": lo, "upper": hi}

    @property
    def midpoint(self):
        return (self.lo + self.hi) / 2


def interval(x):
    return x if isinstance(x, Interval) else Interval.point(x)


def parameter(x, binary_envelope=False):
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise ValueError("invalid numeric parameter")
    exact = F(str(x))
    if not binary_envelope:
        return interval(exact)
    binary = F.from_float(float(x))
    return Interval(min(exact, binary), max(exact, binary))


@lru_cache(maxsize=256)
def _exp_point(x):
    if abs(x) > 4:
        raise ValueError("analytic exp domain exceeds frozen design")
    if x < 0:
        return 1 / _exp_point(-x)
    term = F(1)
    total = term
    for k in range(1, 1000):
        term *= x / k
        total += term
        following = term * x / (k + 1)
        ratio = x / (k + 2)
        if ratio < 1:
            remainder = following / (1 - ratio)
            if remainder < EPS:
                return Interval(total, total + remainder)
    raise ValueError("exponential remainder not certified")


def trig(x, cosine=False):
    """Taylor enclosure, with absolute remainder valid for |x|<=1."""
    x = interval(x)
    if max(abs(x.lo), abs(x.hi)) > 1:
        raise ValueError("trigonometric argument outside frozen domain")
    total = interval(0)
    for k in range(35):
        power = 2 * k + (0 if cosine else 1)
        total += ((-1) ** k) * x**power / math.factorial(power)
    next_power = 70 + (0 if cosine else 1)
    remainder = max(abs(x.lo), abs(x.hi)) ** next_power / math.factorial(next_power)
    return total + Interval(-remainder, remainder)


def probability(x):
    """Intersect a proved probability enclosure with its mathematical domain."""
    x = interval(x)
    if x.hi < 0 or x.lo > 1:
        raise ValueError("probability enclosure misses [0,1]")
    return Interval(max(F(0), x.lo), min(F(1), x.hi))


def binomial_bins(n, eta, parity=False):
    low = [
        interval(math.comb(n, j)) * eta**j * (1 - eta) ** (n - j) if j <= n else interval(0)
        for j in range(5)
    ]
    if parity:
        odd = (1 - (1 - 2 * eta) ** n) / 2
        return low + [
            probability(odd - low[1] - low[3]),
            probability(1 - odd - low[0] - low[2] - low[4]),
        ]
    return low + [probability(1 - sum(low, interval(0)))]


def _coherent_means(case, binary):
    p = case["parameters"]
    alpha = [tuple(parameter(v, binary) for v in z) for z in p["alpha_re_im"]]
    if case["family"] == "coherent_interference":
        theta = parameter(p["beamsplitter_theta"], binary)
        phi = parameter(p["beamsplitter_phi"], binary)
        ct, st, cp, sp = trig(theta, True), trig(theta), trig(phi, True), trig(phi)
        (ar, ai), (br, bi) = alpha
        alpha = [
            (ct * ar - st * (cp * br + sp * bi), ct * ai - st * (cp * bi - sp * br)),
            (ct * br + st * (cp * ar - sp * ai), ct * bi + st * (cp * ai + sp * ar)),
        ]
    return [parameter(eta, binary) * (a * a + b * b) for eta, (a, b) in zip(case["eta"], alpha)]


def _marginal_bins(mean, thermal=False):
    if thermal:
        low = [mean**j / (1 + mean) ** (j + 1) for j in range(5)]
    else:
        vacuum = (-mean).exp()
        low = [vacuum * mean**j / math.factorial(j) for j in range(5)]
    return low + [probability(1 - sum(low, interval(0)))]


def _tensor(marginals):
    return {
        key: math.prod((marginals[j][k] for j, k in enumerate(key)), start=interval(1))
        for key in product(range(6), repeat=len(marginals))
    }


def _nominal_pnr(case, binary):
    """Return semantic PNR representatives and certified probabilities."""
    family = case["family"]
    order = case["output_mode_order"]
    eta = [parameter(v, binary) for v in case["eta"]]
    p = case["parameters"]
    if family == "vacuum":
        return {(j,): interval(int(j == 0)) for j in range(6)}
    if family.startswith("coherent"):
        means = _coherent_means(case, binary)
        return _tensor([_marginal_bins(means[j]) for j in order])
    if family == "thermal":
        mean = parameter(p["mean_photons"], binary) * eta[0]
        return {(j,): v for j, v in enumerate(_marginal_bins(mean, True))}
    if family == "squeezed_vacuum":
        t = parameter(p["tanh_r_squared"], binary)
        weights = [interval(0) for _ in range(7)]
        for k in range(65):
            mass = (1 - t).sqrt() * math.comb(2 * k, k) * (t / 4) ** k
            for j, value in enumerate(binomial_bins(2 * k, eta[0], parity=True)):
                weights[j] += mass * value
        residual = (t**65 / (1 - t).sqrt()).hi
        output = {}
        for j, value in enumerate(weights):
            # Exact ideal support restrictions apply to the whole infinite sum.
            forbidden = (eta[0].hi == 0 and j != 0) or (
                eta[0].lo == eta[0].hi == 1 and j in (1, 3, 5)
            )
            output[(j,)] = (
                interval(0) if forbidden else probability(value + Interval(F(0), residual))
            )
        return output
    if family == "two_mode_squeezed":
        q = parameter(p["q"], binary)
        values = {key: interval(0) for key in product(range(6), repeat=2)}
        for n in range(65):
            bins = [binomial_bins(n, eta[j]) for j in order]
            mass = (1 - q) * q**n
            for key in values:
                values[key] += mass * bins[0][key[0]] * bins[1][key[1]]
        residual = (q**65).hi
        loss_free = all(e.lo == e.hi == 1 for e in eta)
        output = {}
        for key, value in values.items():
            forbidden = (loss_free and key[0] != key[1]) or any(
                eta[physical].hi == 0 and key[column] != 0 for column, physical in enumerate(order)
            )
            output[key] = (
                interval(0) if forbidden else probability(value + Interval(F(0), residual))
            )
        if "unequal_counts" in case["additional_events"]:
            if not loss_free:
                raise ValueError("equality refinement only approved for loss-free TMS")
            output[(5, 6)] = interval(0)
        return output
    raise ValueError("unsupported analytic family")


def cell_key(row, detector):
    return tuple(min(v, 5) if detector == "pnr" else int(v > 0) for v in row)


def indicators(case, row):
    result = ["cell:" + ",".join(map(str, cell_key(row, case["detector"])))]
    if "odd_total_count" in case["additional_events"] and sum(row) % 2:
        result.append("event:odd_total_count")
    if "unequal_counts" in case["additional_events"] and len(set(row)) != 1:
        result.append("event:unequal_counts")
    return result


@dataclass
class Reference:
    semantic: dict
    comparisons: dict
    binary_envelopes: dict

    def certificate(self, budget):
        budget = F(str(budget))
        entries = {}
        for name, value in self.comparisons.items():
            envelope = self.binary_envelopes[name]
            with localcontext() as ctx:
                ctx.prec = 60
                point = str(Decimal(value.midpoint.numerator) / Decimal(value.midpoint.denominator))
            exact_point = F(point)
            error = max(abs(envelope.lo - exact_point), abs(envelope.hi - exact_point))
            if error > budget:
                raise ValueError("reference/parameter error exceeds frozen budget: " + name)
            entries[name] = {
                **value.outward_strings(),
                "binary_literal_envelope": envelope.outward_strings(),
                "reference_point_decimal": point,
                "certified_error_upper": Interval(error, error).outward_strings()["upper"],
            }
        return {
            "method": "rational intervals; exp/trig Taylor remainders; latent-tail bounds",
            "candidate_preparation_certified": False,
            "comparisons": entries,
        }


def _aggregate(case, semantic):
    width = 6 if case["detector"] == "pnr" else 2
    keys = [
        "cell:" + ",".join(map(str, k))
        for k in product(range(width), repeat=len(case["output_mode_order"]))
    ]
    keys += ["event:" + name for name in case["additional_events"]]
    result = {k: interval(0) for k in keys}
    for row, mass in semantic.items():
        for name in indicators(case, row):
            result[name] += mass
    return {name: probability(value) for name, value in result.items()}


@lru_cache(maxsize=128)
def _reference_cached(serialized):
    case = json.loads(serialized)
    nominal = _nominal_pnr(case, False)
    binary = _nominal_pnr(case, True)
    if case["detector"] == "threshold":

        def collapse(table):
            out = {k: interval(0) for k in product(range(2), repeat=len(case["output_mode_order"]))}
            for row, mass in table.items():
                out[tuple(int(v > 0) for v in row)] += mass
            return {k: probability(v) for k, v in out.items()}

        nominal, binary = collapse(nominal), collapse(binary)
    total = sum(nominal.values(), interval(0))
    if not total.lo <= 1 <= total.hi:
        raise ValueError("semantic partition does not certify normalization")
    return Reference(nominal, _aggregate(case, nominal), _aggregate(case, binary))


def reference(case):
    """Compute the independent law for a manifest case, without sampling."""
    keys = ("family", "parameters", "eta", "output_mode_order", "detector", "additional_events")
    selected = {key: case[key] for key in keys}
    if len(selected["output_mode_order"]) not in (1, 2):
        raise ValueError("only one/two measured modes in frozen design")
    return _reference_cached(json.dumps(selected, sort_keys=True, allow_nan=False))


def confidence_radius(K, N, alpha):
    """Outward Bernoulli Hoeffding radius; Decimal ln/sqrt are correctly rounded."""
    if (
        not 1 <= K <= 10000
        or not 1 <= N <= 10000000
        or not F(1, 10**12) <= F(str(alpha)) <= F(1, 2)
    ):
        raise ValueError("invalid confidence design")
    with localcontext() as ctx:
        ctx.prec = 60
        a = Decimal(str(alpha))
        radius = ((Decimal(2 * K) / a).ln() / Decimal(2 * N)).sqrt()
        # With K<=1e4, N<=1e7, alpha>=1e-12: log argument<=2e16,
        # log<=38 and sqrt input>=log(4)/2e7. At 60 digits, combined
        # absolute error after sqrt is <1e-53 (derivative <2000).
        # The 1e-50 allowance also covers decimal division rounding.
        if not 0 < radius < 1:
            raise ValueError("radius outside certified numerical domain")
        return F(radius) + F(1, 10**50)
