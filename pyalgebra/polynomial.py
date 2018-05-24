"""
Implementation of polynomial rings over a field
"""

import copy
import itertools
import operator
import string
import math

from abc import ABC, abstractmethod
from .field import *
from .generic import memoize, CoercableMixin, DivisionError, TotalOrderMixin

RAND_MAX_TERMS = 10
RAND_MAX_DEGREE = 5


class MonomialOrder(ABC):
    """
    Abstract class for Monomial Orders
    """

    def __init__(self, ring):
        self.ring = ring

    @abstractmethod
    def _cmp(self, m1, m2):
        """
        Compare two monomials.

        Returns negative if m1 < m2
        Returns 0 if m1 == m2
        Returns positive if m1 > m2
        """
        pass

    def cmp(self, m1, m2):
        assert m1.ring == self.ring
        assert m2.ring == self.ring
        return self._cmp(m1, m2)


    def __repr__(self):
        return self.__class__.__name__

class Lex(MonomialOrder):
    """
    Lexicographical order
    """

    def _cmp(self, m1, m2):
        diff = map(operator.sub, m1.multidegree, m2.multidegree)
        for e in diff:
            if e != 0:
                return e
        return 0

class Monomial(TotalOrderMixin):
    def __init__(self, ring, multidegree):
        assert isinstance(ring, PolynomialRing)
        self.ring = ring

        assert isinstance(multidegree, list)
        assert len(multidegree) == ring.n
        self.multidegree = multidegree

    def divides(self, other):
        try:
            quotient = other / self
            return True
        except DivisionError as e:
            return False

    def lcm(self, other):
        assert isinstance(other, self.__class__)
        multidegree = map(max, self.multidegree, other.multidegree)
        return Monomial(self.ring, list(multidegree))

    def __mul__(self, other):
        assert isinstance(other, self.__class__)
        assert other.ring == other.ring

        new = list(map(operator.add, self.multidegree, other.multidegree))
        return Monomial(self.ring, new)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        quotient = list(map(operator.sub, self.multidegree, other.multidegree))
        if any(q < 0 for q in quotient):
            raise DivisionError()
        return Monomial(self.ring, quotient)

    def __cmp__(self, other):
        return self.ring.order.cmp(self, other)

    def __hash__(self):
        return hash((self.ring, tuple(self.multidegree)))

    def __repr__(self):
        if self == self.ring.one_monomial():
            return f"{self.ring.field.one()}"

        vars_rep = []
        for v, d in zip(self.ring.variables, self.multidegree):
            if d == 0:
                continue
            if d == 1:
                vars_rep.append(f"{v}")
            else:
                vars_rep.append(f"{v}**{d}")
        return " * ".join(vars_rep)

    __str__ = __repr__


class Term(CoercableMixin, TotalOrderMixin):
    def __init__(self, coefficient, monomial):
        assert isinstance(monomial, Monomial)
        self.monomial = monomial
        self.ring = monomial.ring
        coefficient = self.ring.field.coerce(coefficient)
        assert coefficient != self.ring.field.zero()
        self.coefficient = coefficient

    def coerce(self, elt):
        if isinstance(elt, self.__class__):
            return elt
        if isinstance(elt, Monomial):
            return Term(self.ring.field.one(), elt)
        c = self.ring.field.coerce(elt)
        return Term(c, self.ring.one_monomial())

    def divides(self, other):
        other = self.coerce(other)
        return self.monomial.divides(other.monomial)

    def op_mul(self, other):
        return Term(self.coefficient * other.coefficient,
                    self.monomial * other.monomial)

    def op_add(self, other):
        return Term(self.coefficient + other.coefficient, self.monomial)

    def op_truediv(self, other):
        c = self.coefficient / other.coefficient
        m = self.monomial / other.monomial
        return Term(c, m)

    def op_cmp(self, other):
        res = self.monomial.__cmp__(other.monomial)
        if res == 0:
            return self.coefficient - other.coefficient
        return res

    def __neg__(self):
        return Term(-self.coefficient, self.monomial)

    def __repr__(self):
        if self.monomial == self.ring.one_monomial():
            return f"{self.coefficient}"
        elif self.coefficient == self.ring.field.one():
            return f"{self.monomial}"
        elif -self.coefficient == self.ring.field.one():
            return f"-{self.monomial}"
        else:
            return f"{self.coefficient}*{self.monomial}"

    __str__ = __repr__

class Polynomial(CoercableMixin):
    def __init__(self, ring, terms):
        self.ring = ring

        assert isinstance(terms, list)
        self.terms = terms

        # TODO: should we reduce lazily instead?
        self._normal_form()

    def _normal_form(self):
        """
        Convert the terms of `self` to normal form. Combine terms with equal
        monomials, remove zero terms, and sort according to the monomial order.
        """
        coefficient_map = {}
        for t in self.terms:
            c, m = t.coefficient, t.monomial
            if c == 0:
                continue
            if m in coefficient_map:
                coefficient_map[m] += c
            else:
                coefficient_map[m] = c
        self.terms = [Term(c, m) for m, c in coefficient_map.items() if c != 0]
        self.terms = sorted(self.terms, reverse=True)

    def coerce(self, other):
        return self.ring.coerce(other)

    def is_zero(self):
        return len(self.terms) == 0

    def lt(self):
        """
        Return the leading term of self`
        """
        # TODO: what about the zero polynomial
        return self.terms[0]

    def lc(self):
        """
        Return the leading coefficient of `self`
        """
        return self.lt().coefficient

    def lm(self):
        """
        Return the leading monomial of `self`
        """
        return self.lt().monomial

    def degree(self):
        """
        Return the (multi)degree of `self`
        """
        return self.lm().multidegree

    def monic(self):
        """
        Return the monic version of `self`
        """
        c = self.lc()
        return Polynomial(self.ring, [t / c for t in self.terms])

    def divide_many(self, others):
        """
        Perform the division algorithm with `self` by the ordered set of
        polynomials `others`.

        Returns (qs, r) such that sum(map(operator.mul, qs, others)) + r ==
        self and not r.lt().divides(o.lt()) for all o in others.

        The result is guaranteed to be unique for the given order of `others`.
        It is unique for all orders of `others` if and only if `others` is a
        groebner basis for their ideal.
        """
        zero = self.ring.zero()
        qs = [zero for _ in others]
        r = zero
        p = self
        while p != zero:
            # Division invariant.
            assert sum(map(operator.mul, qs, others)) + p + r == self

            # Try to divide by a polynomial.
            for i, o in enumerate(others):
                try:
                    t = p.lt() / o.lt()
                except DivisionError:
                    # If the terms were not divisible, try the next polynomial.
                    continue

                # Perform the division.
                qs[i] += t
                p -= t * o
                break
            else:
                # None of them divided. Move the leading term to r.
                r += p.lt()
                p -= p.lt()

        assert sum(map(operator.mul, qs, others)) + r == self
        return (qs, r)

    def divide(self, other):
        """
        Perform the division algorithm with `self` and `other`

        Returns q and r such that q*other + r == self and
        (r == 0 or not r.lm().divides(q.lm()))
        """
        (q,), r = self.divide_many([other])
        return (q, r)

    def divides(self, other):
        """
        Returns true if `self` divides `other`
        """
        other = self.coerce(other)
        return other.divide(self)[1] == self.ring.zero()

    def gcd(self, other):
        """
        Returns the gcd of `self` and `other`.

        Simple Euclidean Algorithm
        """
        assert self.ring.n == 1, "Only works with univariate rings"
        zero = self.ring.zero()
        h = self
        s = other
        while s != zero:
            r = h % s
            h = s
            s = r
        return h

    def op_add(self, other):
        return Polynomial(self.ring, self.terms + other.terms)

    def op_sub(self, other):
        return Polynomial(self.ring, self.terms + (-other).terms)

    def op_mul(self, other):
        # TODO: optimize this using karatsuba
        assert self.ring == other.ring
        pairs = itertools.product(self.terms, other.terms)
        raw_terms = [operator.mul(a, b) for a,b in pairs]
        return Polynomial(self.ring, raw_terms)

    def __pow__(self, exp):
        assert exp >= 0
        res = self.ring.one()
        i = math.ceil(math.log(exp, 2))
        while i >= 0:
            res *= res
            if exp & (1 << i):
                res *= self
            i -= 1
        return res

    def op_truediv(self, other):
        q, r = self.divide(other)
        if r != 0:
            raise DivisionError("Remainder is nonzero")
        return q

    def op_mod(self, other):
        return self.divide(other)[1]

    def __eq__(self, other):
        other = self.coerce(other)
        if self.terms == other.terms:
            return 1
        return 0

    def __neg__(self):
        return Polynomial(self.ring, [-t for t in self.terms])

    def __repr__(self):
        if len(self.terms) == 0:
            return f"{self.ring.field.zero()}"
        res = repr(self.terms[0])
        for term in self.terms[1:]:
            if term.coefficient < 0:
                res += f" - {-term}"
            else:
                res += f" + {term}"
        return res

    __str__ = __repr__


class PolynomialRingIdeal:
    def __init__(self, ring, *gens):
        assert isinstance(ring, PolynomialRing)
        self.ring = ring

        assert len(gens) > 0, "Need at least one generator"
        assert all(isinstance(g, Polynomial) and g.ring == ring for g in gens)
        self.gens = list(gens)

    @memoize
    def groebner_basis(self):
        """
        Return the unique reduced Groebner basis for `self`.

        Uses Buchberger's algorithm to build a Groebner basis, then minimizes
        and reduces the basis. This is not a high-performance implementation.
        """
        def spoly(f, g):
            num = f.lm().lcm(g.lm())
            ft = num / f.lt()
            gt = num / g.lt()
            return ft * f - gt * g

        # Build a Groebner basis using Buchberger's algorithm.
        groebner = [g.monic() for g in self.gens]
        new_pairs = itertools.combinations(groebner, 2)
        while True:
            new = []
            for f, g in new_pairs:
                s = spoly(f, g).divide_many(groebner)[1]
                if s == 0:
                    continue
                s = s.monic()
                if s not in new and s not in groebner:
                    new.append(s.monic())

            # We've stabilized.
            if len(new) == 0:
                break

            new_pairs = itertools.chain(
                itertools.product(groebner, new),
                itertools.combinations(new, 2)
            )
            groebner += new

        # Minimize it.
        lts = [self.ring.coerce(g.lt()) for g in groebner]
        while True:
            for i in range(len(lts)):
                lt = lts[i]
                others = lts[:i] + lts[i+1:]
                if lt.divide_many(others)[1] == self.ring.zero():
                    lts = others
                    groebner = groebner[:i] + groebner[i+1:]
                    break
            else:
                break

        # Reduce it.
        for i in range(len(groebner)):
            g = groebner[i]
            others = groebner[:i] + groebner[i+1:]
            g = g.divide_many(others)[1]
            groebner[i] = g

        # Sort it.
        return sorted(groebner, key=lambda g:g.lm())

    def __rmod__(self, other):
        """
        Returns the unique remainder of `other` into `self`.
        """
        other = self.gens[0].coerce(other) # Coerce `other` into a polynomial.
        return other.divide_many(self.groebner_basis())[1]

    def __contains__(self, other):
        return other % self == self.ring.zero()

    def __mul__(self, other):
        gens = [a * b for a,b in itertools.product(self.gens, other.gens)]
        return PolynomialRingIdeal(self.ring, gens)

    def __add__(self, other):
        return PolynomialRingIdeal(self.ring, self.gens + other.gens)

    def __hash__(self):
        # TODO: Ideally we would hash our groebner basis, but we can't @memoize
        # hashes its arguments. For now, the hash is simply id(self).
        return id(self)

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        return self.groebner_basis() == other.groebner_basis()

    def __repr__(self):
        return (f"Ideal ({', '.join(repr(g) for g in self.gens)}) "
                f"in {self.ring}")

    __str__ = __repr__


class PolynomialRing(Ring):
    DEFAULT_VARS = ['x', 'y', 'z']

    #TODO: dynamically make a specific class for each ring
    element_class = Polynomial

    def __init__(self, field, n=1, variables=None, order=Lex):
        """
        Constructs a polynonial ring in n variables using
        """
        assert isinstance(field, Field)
        self.field = field

        assert isinstance(n, int)
        self.n = n

        if variables is not None:
            assert len(variables) == n
            for v in variables:
                assert isinstance(v, str)
                assert len(v) > 0
                assert v[0] in string.ascii_letters
        elif n <= len(self.DEFAULT_VARS):
            variables = self.DEFAULT_VARS[:n]
        else:
            l = math.ceil(math.log(n, 10))
            variables = ['x_{:0{l}d}'.format(i, l=l) for i in range(n)]
        self.variables = variables

        assert issubclass(order, MonomialOrder)
        self.order = order(self)

    def zero(self):
        return Polynomial(self, [])

    def one_monomial(self):
        zero = self.field.zero()
        return Monomial(self, [zero]*self.n)

    def one(self):
        one = self.field.one()
        return Polynomial(self, [Term(one, self.one_monomial())])

    def coerce(self, elt):
        if isinstance(elt, self.element_class):
            return elt
        if isinstance(elt, Term) and elt.ring == self:
            return Polynomial(self, [elt])
        if isinstance(elt, Monomial) and elt.ring == self:
            return Polynomial(self, [Term(self.ring.one(), elt)])
        c = self.field.coerce(elt)
        if c == self.field.zero():
            return self.zero()
        return Polynomial(self, [Term(c, self.one_monomial())])

    def gens(self):
        one = self.field.one()
        zero = self.field.zero()
        mds = [[zero]*i + [one] + [zero]*(self.n - i - 1)
               for i in range(self.n)]
        return [Polynomial(self, [Term(one, Monomial(self, md))]) for md in
                mds]

    def an_element(self):
        return self.gens()[0]

    def random_element(self, max_degree=RAND_MAX_DEGREE,
                       max_terms=RAND_MAX_TERMS, monic=False, **kwargs):
        def random_term():
            c = self.field.random_element(**kwargs)
            while c == 0:
                c = self.field.random_element(**kwargs)
            md = [random.randint(0, max_degree) for _ in range(self.n)]
            return Term(c, Monomial(self, md))
        num_terms = random.randint(1, max_terms)
        terms = [random_term() for _ in range(num_terms)]
        p = Polynomial(self, terms)
        if monic:
            p.terms[0].coefficient = self.field.one()
        return p

    def order(self):
        return math.inf

    def ideal(self, *gens):
        return PolynomialRingIdeal(self, *gens)

    def __repr__(self):
        return (f"PolynomialRing"
                f"{(self.field, self.n, self.variables, self.order)}")

    __str__ = __repr__
