#!/usr/bin/env python3
"""mub_gen.py -- generate mutually unbiased bases (MUBs) in any dimension.

A set of orthonormal bases of C^D is mutually unbiased if, for any two
vectors drawn from two different bases, |<u|v>|^2 = 1/D. For D a prime
power a COMPLETE set of D+1 MUBs exists and is produced here by the
Desarguesian-spread (Heisenberg-Weyl) construction over the finite field
GF(D). For D not a prime power only a partial set is known; this code
returns the best construction available from the prime-power factors
(min over factors of (q_i + 1) bases) and flags the gap.

Author-education tool for the QCE scaling effort. Public domain for the
group's use.
"""
import sys
import numpy as np
from itertools import product


# --------------------------------------------------------------------- #
#  Finite field GF(p^n)                                                  #
# --------------------------------------------------------------------- #
# Elements are integers 0..p^n-1 read as base-p digit vectors = poly
# coefficients (low-order first). Arithmetic is mod a chosen monic
# irreducible polynomial of degree n over GF(p).

# Minimal irreducible polynomials (coeffs low->high, monic implied top)
# stored as the LOW coefficients c0..c_{n-1} of  x^n + c_{n-1} x^{n-1}
# + ... + c0.  Verified irreducible for the (p,n) needed here.
_IRRED = {
    (2, 2): [1, 1],          # x^2 + x + 1      4
    (2, 3): [1, 1, 0],       # x^3 + x + 1      8
    (2, 4): [1, 1, 0, 0],    # x^4 + x + 1      16
    (2, 5): [1, 0, 1, 0, 0], # x^5 + x^2 + 1    32
    (3, 2): [1, 0],          # x^2 + 1          9
    (3, 3): [1, 2, 0],       # x^3 + 2x + 1     27
    (5, 2): [2, 0],          # x^2 + 2          25
    (7, 2): [1, 0],          # x^2 + 1          49
}

# Chekcs to see if the number can be broken down to p^n
def _factor_prime_power(D):
    """Return (p, n) if D = p^n is a prime power, else None."""
    for p in range(2, D + 1):
        if D % p == 0:
            n, q = 0, D
            while q % p == 0:
                q //= p
                n += 1
            return (p, n) if q == 1 else None
    return None


class GF:
    """Finite field GF(p^n) with element trace to GF(p)."""

    def __init__(self, p, n):
        self.p, self.n, self.q = p, n, p ** n
        if n == 1:
            self.irr = None
        else:
            if (p, n) not in _IRRED:
                raise ValueError(f"no irreducible poly tabulated for "
                                 f"GF({p}^{n}); add one to _IRRED")
            self.irr = _IRRED[(p, n)]
        # precompute trace of every element
        self._tr = [self._trace(x) for x in range(self.q)]

    def _digits(self, x):
        d = []
        for _ in range(self.n):
            d.append(x % self.p)
            x //= self.p
        return d

    def _from_digits(self, d):
        x = 0
        for c in reversed(d):
            x = x * self.p + (c % self.p)
        return x

    def add(self, x, y):
        dx, dy = self._digits(x), self._digits(y)
        return self._from_digits([(a + b) % self.p for a, b in zip(dx, dy)])

    def neg(self, x):
        return self._from_digits([(-a) % self.p for a in self._digits(x)])

    def mul(self, x, y):
        if self.n == 1:
            return (x * y) % self.p
        # polynomial multiply then reduce mod irr
        a, b = self._digits(x), self._digits(y)
        prod = [0] * (2 * self.n)
        for i in range(self.n):
            for j in range(self.n):
                prod[i + j] = (prod[i + j] + a[i] * b[j]) % self.p
        # reduce degree >= n using x^n = -(c_{n-1} x^{n-1}+...+c0)
        for deg in range(2 * self.n - 1, self.n - 1, -1):
            coef = prod[deg]
            if coef:
                prod[deg] = 0
                for k in range(self.n):
                    prod[deg - self.n + k] = (
                        prod[deg - self.n + k]
                        - coef * self.irr[k]) % self.p
        return self._from_digits(prod[:self.n])

    def _trace(self, x):
        # tr(x) = x + x^p + x^{p^2} + ... + x^{p^{n-1}}  (lands in GF(p))
        s, xp = 0, x
        for _ in range(self.n):
            s = self.add(s, xp)
            # xp <- xp^p
            r = 1
            for _ in range(self.p):
                r = self.mul(r, xp)
            xp = r
        # s is an element of GF(p) embedded in GF(p^n): its 0th digit
        return self._digits(s)[0]

    def trace(self, x):
        return self._tr[x]


# --------------------------------------------------------------------- #
#  Heisenberg-Weyl displacement operators over GF(q)                     #
# --------------------------------------------------------------------- #
def _hw_operators(F):
    """X_a (additive shift) and Z_b (character phase) for all a,b in GF."""
    q, p = F.q, F.p
    w = np.exp(2j * np.pi / p)
    X = {}
    Z = {}
    for a in range(q):
        Xa = np.zeros((q, q), dtype=complex)
        for g in range(q):
            Xa[F.add(g, a), g] = 1.0
        X[a] = Xa
    for b in range(q):
        Z[b] = np.diag([w ** F.trace(F.mul(b, g)) for g in range(q)])
    return X, Z


def _basis_from_line(F, X, Z, slope, rng):
    """Common eigenbasis of the maximal commuting class on a line."""
    q = F.q
    if slope == 'inf':
        # vertical line {Z_b}: all diagonal -> computational basis
        return np.eye(q, dtype=complex)
    # line {X_a Z_{slope*a}}: the operators commute and are unitary, so
    # A = sum_a gamma_a M_a is normal; H = A + A^dagger is Hermitian and
    # shares their eigenbasis. COMPLEX gamma_a are required: real weights
    # leave conjugate eigenvalue pairs degenerate (e.g. the shift/Fourier
    # line), which would let eigh return an arbitrary basis.
    A = np.zeros((q, q), dtype=complex)
    for a in range(1, q):
        b = F.mul(slope, a)
        M = X[a] @ Z[b]
        gamma = rng.normal() + 1j * rng.normal()
        A = A + gamma * M
    H = A + A.conj().T
    w_, V = np.linalg.eigh(H)
    return V


def mubs_prime_power(p, n, seed=0):
    """Complete set of D+1 MUBs for D = p^n."""
    F = GF(p, n)
    X, Z = _hw_operators(F)
    rng = np.random.default_rng(seed)
    bases = [_basis_from_line(F, X, Z, 'inf', rng)]           # computational
    for slope in range(F.q):
        bases.append(_basis_from_line(F, X, Z, slope, rng))
    return bases


def _factor(D):
    fac, d = {}, D
    p = 2
    while p * p <= d:
        while d % p == 0:
            fac[p] = fac.get(p, 0) + 1
            d //= p
        p += 1
    if d > 1:
        fac[d] = fac.get(d, 0) + 1
    return fac


def mubs(D, seed=0):
    """Return (list_of_bases, info). Each basis is a (D,D) unitary whose
    COLUMNS are the basis vectors."""
    pp = _factor_prime_power(D)
    if pp:
        p, n = pp
        B = mubs_prime_power(p, n, seed)
        return B, dict(prime_power=True, count=len(B), maximal=D + 1)
    # composite non-prime-power: tensor MUBs of prime-power factors
    fac = _factor(D)
    factor_sets = []
    for p, n in fac.items():
        factor_sets.append(mubs_prime_power(p, n, seed))
    K = min(len(s) for s in factor_sets)          # bases we can tensor
    bases = []
    for k in range(K):
        B = factor_sets[0][k]
        for s in factor_sets[1:]:
            B = np.kron(B, s[k])
        bases.append(B)
    return bases, dict(prime_power=False, count=K, maximal='unknown',
                       note=f"D={D} is not a prime power; a complete set "
                            f"is not known. {K} MUBs constructed by the "
                            f"tensor-product bound.")


# --------------------------------------------------------------------- #
#  Verification                                                          #
# --------------------------------------------------------------------- #
def verify(bases, D, tol=1e-9):
    ok = True
    # orthonormality within each basis
    for B in bases:
        if np.linalg.norm(B.conj().T @ B - np.eye(D)) > tol:
            ok = False
    # unbiasedness across bases
    worst = 0.0
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            M = np.abs(bases[i].conj().T @ bases[j]) ** 2
            worst = max(worst, np.abs(M - 1.0 / D).max())
    return ok and worst < tol, worst


def _fmt(z):
    r, im = z.real, z.imag
    def s(x):
        if abs(x) < 1e-9:
            return "0"
        if abs(x - round(x)) < 1e-6:
            return f"{round(x):+d}"
        return f"{x:+.4f}"
    if abs(im) < 1e-9:
        return f"{s(r)}"
    if abs(r) < 1e-9:
        return f"{s(im)}i"
    return f"{s(r)}{s(im)}i"


def main():
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    show = '--show' in sys.argv
    B, info = mubs(D)
    good, worst = verify(B, D)
    print(f"D = {D}")
    print(f"  prime power : {info['prime_power']}")
    print(f"  MUBs built  : {info['count']}   (maximal possible: "
          f"{info['maximal']})")
    if 'note' in info:
        print(f"  note        : {info['note']}")
    print(f"  verified    : {good}   (max |<.|.>|^2 - 1/D deviation "
          f"= {worst:.2e})")
    if show:
        scale = np.sqrt(D)
        for k, basis in enumerate(B):
            print(f"\n  Basis {k}  (columns are vectors; entries x {1}/"
                  f"sqrt({D})):")
            M = basis * scale
            for col in range(D):
                v = M[:, col]
                print("    [ " + "  ".join(_fmt(z) for z in v) + " ]")


if __name__ == "__main__":
    main()
