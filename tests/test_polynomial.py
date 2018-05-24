from pyalgebra import *

R = PolynomialRing(QQ(), 3)
x,y,z = R.gens()

def random_element():
    return R.random_element(max_degree=3, max_terms=2, monic=True, integral=True)

def test_divide():
    p1 = random_element()
    p2 = random_element()

    r = random_element()
    while r.lt().divides(p1.lt()):
        r = random_element()

    f = p1*p2 + r
    q, rp = f.divide(p1)
    assert f == q*p1 + rp
    assert rp.lt() == r.lt()

def test_groebner():
    p1 = random_element()
    p2 = random_element()

    I = R.ideal(p1, p2)

    assert p1 in I
    assert p2 in I

    g = I.groebner_basis()
    G = R.ideal(*g)

    assert G == I
    assert G.groebner_basis() == g

def test_minpoly_solve():
    R = PolynomialRing(QQ(), 4, ['x0', 'x1', 'x2', 'z'])
    x0, x1, x2, z = R.gens()
    I = R.ideal(x0*z + 1 - x1, x0*z - 1 - x2, x2*z - x1, x1*z - x0)
    g = I.groebner_basis()

    assert g[0] == z**2 + z + Fraction(1,2)
