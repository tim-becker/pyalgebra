import math
import numbers
import random

from abc import ABC, abstractmethod, abstractproperty
from fractions import Fraction

from .ring import Ring
from .generic import CoercionError

RAND_MAX_INT = 2**16

class Field(Ring, ABC):
    @abstractmethod
    def characteristic(self):
        """
        Return the characteristic of the field
        """
        pass


class QQ(Field):
    element_class = Fraction

    def zero(self):
        return Fraction(0, 1)

    def one(self):
        return Fraction(1, 1)

    def an_element(self):
        return self.one()

    def coerce(self, elt):
        if isinstance(elt, self.element_class):
            return elt
        elif isinstance(elt, numbers.Number):
            return self.one() * elt
        else:
            raise CoercionError("Cannot coerce {} to {}".format(
              elt, self))

    def random_element(self, max_int=RAND_MAX_INT, integral=True):
        return Fraction(random.randint(0, max_int),
                        random.randint(1, 1 if integral else max_int))

    def order(self):
        return math.inf

    def characteristic(self):
        return 0

    def __repr__(self):
        return "QQ()"
