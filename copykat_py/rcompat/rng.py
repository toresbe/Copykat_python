"""R's default random number generator, draw for draw.

A line-by-line port of R 4.x ``src/main/RNG.c`` (Mersenne-Twister and ``set.seed``),
``src/nmath/snorm.c`` (``norm_rand``, INVERSION), ``src/nmath/qnorm.c``,
``src/nmath/sexp.c`` (``exp_rand``) and ``src/nmath/rgamma.c``. After
``RRandom(1234)`` the draws equal those of ``set.seed(1234)`` in R with the default
``RNGkind()`` ("Mersenne-Twister", "Inversion"). ``tests/test_rcompat_rng.py``
checks this against values printed by R.

The state is two small arrays so that the numba kernels can advance it in place.
"""

import math

import numpy as np
from numba import njit

_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER_MASK = np.uint32(0x80000000)
_LOWER_MASK = np.uint32(0x7FFFFFFF)
_I2_32M1 = 2.328306437080797e-10  # 1/(2^32 - 1)


def _seed_state(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``set.seed(seed)``: RNG_Init's scrambling, then FixupSeeds (mti = 624)."""
    s = seed & 0xFFFFFFFF
    for _ in range(50):
        s = (69069 * s + 1) & 0xFFFFFFFF
    i_seed = np.empty(_N + 1, dtype=np.uint32)
    for j in range(_N + 1):
        s = (69069 * s + 1) & 0xFFFFFFFF
        i_seed[j] = s
    mt = i_seed[1:].copy()
    mti = np.array([_N], dtype=np.int64)  # dummy[0], forced to 624 by FixupSeeds(initial=1)
    return mt, mti


@njit(cache=False)
def _mt_genrand(mt: np.ndarray, mti: np.ndarray) -> float:
    if mti[0] >= _N:
        for kk in range(_N - _M):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + _M] ^ (y >> np.uint32(1)) ^ (_MATRIX_A if (y & np.uint32(1)) else np.uint32(0))
        for kk in range(_N - _M, _N - 1):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + (_M - _N)] ^ (y >> np.uint32(1)) ^ (_MATRIX_A if (y & np.uint32(1)) else np.uint32(0))
        y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
        mt[_N - 1] = mt[_M - 1] ^ (y >> np.uint32(1)) ^ (_MATRIX_A if (y & np.uint32(1)) else np.uint32(0))
        mti[0] = 0
    y = mt[mti[0]]
    mti[0] += 1
    y ^= y >> np.uint32(11)
    y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
    y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
    y ^= y >> np.uint32(18)
    return float(y) * 2.3283064365386963e-10


@njit(cache=False)
def unif_rand(mt: np.ndarray, mti: np.ndarray) -> float:
    x = _mt_genrand(mt, mti)
    if x <= 0.0:
        return 0.5 * _I2_32M1
    if (1.0 - x) <= 0.0:
        return 1.0 - 0.5 * _I2_32M1
    return x


@njit(cache=False)
def _qnorm(p: float) -> float:
    """qnorm5(p, 0, 1, lower_tail=TRUE, log_p=FALSE): Wichura's AS241, as in R."""
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return (
            q
            * (
                (
                    (
                        (
                            (
                                ((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                                + 45921.953931549871457
                            )
                            * r
                            + 13731.693765509461125
                        )
                        * r
                        + 1971.5909503065514427
                    )
                    * r
                    + 133.14166789178437745
                )
                * r
                + 3.387132872796366608
            )
            / (
                (
                    (
                        (
                            (
                                ((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r
                                + 21213.794301586595867
                            )
                            * r
                            + 5394.1960214247511077
                        )
                        * r
                        + 687.1870074920579083
                    )
                    * r
                    + 42.313330701600911252
                )
                * r
                + 1.0
            )
        )
    lp = math.log((0.5 - p + 0.5) if q > 0 else p)
    r = math.sqrt(-lp)
    if r <= 5.0:
        r += -1.6
        val = (
            (
                (
                    (
                        (
                            ((r * 7.7454501427834140764e-4 + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                            + 1.27045825245236838258
                        )
                        * r
                        + 3.64784832476320460504
                    )
                    * r
                    + 5.7694972214606914055
                )
                * r
                + 4.6303378461565452959
            )
            * r
            + 1.42343711074968357734
        ) / (
            (
                (
                    (
                        (
                            ((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + 0.0151986665636164571966)
                            * r
                            + 0.14810397642748007459
                        )
                        * r
                        + 0.68976733498510000455
                    )
                    * r
                    + 1.6763848301838038494
                )
                * r
                + 2.05319162663775882187
            )
            * r
            + 1.0
        )
    else:
        # r <= 27 always holds here: p >= 2^-27 * 2^-32 from norm_rand.
        r += -5.0
        val = (
            (
                (
                    (
                        (
                            ((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + 0.0012426609473880784386)
                            * r
                            + 0.026532189526576123093
                        )
                        * r
                        + 0.29656057182850489123
                    )
                    * r
                    + 1.7848265399172913358
                )
                * r
                + 5.4637849111641143699
            )
            * r
            + 6.6579046435011037772
        ) / (
            (
                (
                    (
                        (
                            ((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5)
                            * r
                            + 7.868691311456132591e-4
                        )
                        * r
                        + 0.0148753612908506148525
                    )
                    * r
                    + 0.13692988092273580531
                )
                * r
                + 0.59983220655588793769
            )
            * r
            + 1.0
        )
    if q < 0.0:
        val = -val
    return val


@njit(cache=False)
def norm_rand(mt: np.ndarray, mti: np.ndarray) -> float:
    big = 134217728.0  # 2^27
    u1 = unif_rand(mt, mti)
    u1 = float(int(big * u1)) + unif_rand(mt, mti)
    return _qnorm(u1 / big)


_EXP_Q = np.array(
    [
        0.6931471805599453,
        0.9333736875190459,
        0.9888777961838675,
        0.9984959252914960,
        0.9998292811061389,
        0.9999833164100727,
        0.9999985691438767,
        0.9999998906925558,
        0.9999999924734159,
        0.9999999995283275,
        0.9999999999728814,
        0.9999999999985598,
        0.9999999999999289,
        0.9999999999999968,
        0.9999999999999999,
        1.0000000000000000,
    ]
)


@njit(cache=False)
def exp_rand(mt: np.ndarray, mti: np.ndarray) -> float:
    q = _EXP_Q
    a = 0.0
    u = unif_rand(mt, mti)
    while u <= 0.0 or u >= 1.0:
        u = unif_rand(mt, mti)
    while True:
        u += u
        if u > 1.0:
            break
        a += q[0]
    u -= 1.0
    if u <= q[0]:
        return a + u
    i = 0
    ustar = unif_rand(mt, mti)
    umin = ustar
    while True:
        ustar = unif_rand(mt, mti)
        if umin > ustar:
            umin = ustar
        i += 1
        if not (u > q[i]):
            break
    return a + umin * q[0]


@njit(cache=False)
def rgamma_one(mt: np.ndarray, mti: np.ndarray, a: float, scale: float) -> float:
    """One draw of R's ``rgamma(1, shape = a, scale = scale)`` (GS for a < 1, GD otherwise)."""
    sqrt32 = 5.656854
    exp_m1 = 0.36787944117144232159
    q1, q2, q3, q4 = 0.04166669, 0.02083148, 0.00801191, 0.00144121
    q5, q6, q7 = -7.388e-5, 2.4511e-4, 2.424e-4
    a1, a2, a3, a4 = 0.3333333, -0.250003, 0.2000062, -0.1662921
    a5, a6, a7 = 0.1423657, -0.1367177, 0.1233795

    if a <= 0.0 or scale <= 0.0:
        return 0.0 if (scale == 0.0 or a == 0.0) else math.nan
    if a < 1.0:
        e = 1.0 + exp_m1 * a
        while True:
            p = e * unif_rand(mt, mti)
            if p >= 1.0:
                x = -math.log((e - p) / a)
                if exp_rand(mt, mti) >= (1.0 - a) * math.log(x):
                    break
            else:
                x = math.exp(math.log(p) / a)
                if exp_rand(mt, mti) >= x:
                    break
        return scale * x

    # GD algorithm. R caches s2, s, d, q0, b, si, c between calls with the same a;
    # recomputing them gives the same values.
    s2 = a - 0.5
    s = math.sqrt(s2)
    d = sqrt32 - s * 12.0

    t = norm_rand(mt, mti)
    x = s + 0.5 * t
    ret_val = x * x
    if t >= 0.0:
        return scale * ret_val

    u = unif_rand(mt, mti)
    if d * u <= t * t * t:
        return scale * ret_val

    r = 1.0 / a
    q0 = ((((((q7 * r + q6) * r + q5) * r + q4) * r + q3) * r + q2) * r + q1) * r
    if a <= 3.686:
        b = 0.463 + s + 0.178 * s2
        si = 1.235
        c = 0.195 / s - 0.079 + 0.16 * s
    elif a <= 13.022:
        b = 1.654 + 0.0076 * s2
        si = 1.68 / s + 0.275
        c = 0.062 / s + 0.024
    else:
        b = 1.77
        si = 0.75
        c = 0.1515 / s

    if x > 0.0:
        v = t / (s + s)
        if abs(v) <= 0.25:
            q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v
        else:
            q = q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)
        if math.log(1.0 - u) <= q:
            return scale * ret_val

    while True:
        e = exp_rand(mt, mti)
        u = unif_rand(mt, mti)
        u = u + u - 1.0
        if u < 0.0:
            t = b - si * e
        else:
            t = b + si * e
        if t >= -0.71874483771719:
            v = t / (s + s)
            if abs(v) <= 0.25:
                q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v
            else:
                q = q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)
            if q > 0.0:
                w = math.expm1(q)
                if c * abs(u) <= w * math.exp(e - 0.5 * t * t):
                    break
    x = s + 0.5 * t
    return scale * x * x


@njit(cache=False)
def _rgamma_fill(mt: np.ndarray, mti: np.ndarray, a: float, scale: float, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = rgamma_one(mt, mti, a, scale)


@njit(cache=False)
def _runif_fill(mt: np.ndarray, mti: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = unif_rand(mt, mti)


@njit(cache=False)
def _rnorm_fill(mt: np.ndarray, mti: np.ndarray, mean: np.ndarray, sd: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.shape[0]):
        out[i] = mean[i % mean.shape[0]] + sd[i % sd.shape[0]] * norm_rand(mt, mti)


class RRandom:
    """An R random-number stream: ``RRandom(seed)`` is ``set.seed(seed)``."""

    def __init__(self, seed: int) -> None:
        self.mt, self.mti = _seed_state(seed)

    def runif(self, n: int) -> np.ndarray:
        out = np.empty(n)
        _runif_fill(self.mt, self.mti, out)
        return out

    def rnorm(self, n: int, mean: float | np.ndarray = 0.0, sd: float | np.ndarray = 1.0) -> np.ndarray:
        out = np.empty(n)
        _rnorm_fill(
            self.mt,
            self.mti,
            np.atleast_1d(np.asarray(mean, dtype=np.float64)),
            np.atleast_1d(np.asarray(sd, dtype=np.float64)),
            out,
        )
        return out

    def rgamma(self, n: int, shape: float, rate: float = 1.0) -> np.ndarray:
        """``rgamma(n, shape, rate)``; R passes ``scale = 1/rate`` to the C code."""
        out = np.empty(n)
        _rgamma_fill(self.mt, self.mti, float(shape), 1.0 / float(rate), out)
        return out
