# pyalgebra

Toy computer algebra system in pure python.

Note: This code is entirely experimental and is not written to be performant.
I'm implementing this for fun and practice.

# Features

The only interesting thing implemented so far is multivariate polynomial rings.

The implementation would probably work over arbitrary fields, but currently
the only implemented field is the rationals.

# Example

```python
from pyalgebra import *

R = PolynomialRing(QQ(), 4, ['x0', 'x1', 'x2', 'z'])
x0, x1, x2, z = R.gens()

I = R.ideal(x0*z + 1 - x1, x0*z - 1 - x2, x2*z - x1, x1*z - x0)
print(I.groebner_basis())
```
would output
```
[z**2 + z + 1/2, x2 + 4/5*z + 8/5, x1 + 4/5*z - 2/5, x0 - 6/5*z - 2/5]
```
