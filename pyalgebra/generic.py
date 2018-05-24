"""
Generic classes / exceptions
"""

import functools

from abc import ABC, abstractmethod

memoize = functools.lru_cache(maxsize=1)

class DivisionError(Exception):
    """
    Errors occuring during division
    """

class CoercionError(Exception):
    """
    Errors occuring during coercion
    """

@functools.total_ordering
class TotalOrderMixin(ABC):
    """
    Mixin for a total order from a __cmp__ method
    """

    @abstractmethod
    def __cmp__(self, other):
        pass

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

#TODO: Improve coercion and expand operator support
class CoercableMixin(ABC):
    """
    Mixin for operator overloading with coercion
    """
    @abstractmethod
    def coerce(self, other):
        pass

    def op_add(self, other):
        return NotImplemented

    def __add__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return self.op_add(other)

    def __radd__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return other.__add__(self)

    def op_sub(self, other):
        return NotImplemented

    def __sub__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return self.op_sub(other)

    def __rsub__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return other.__sub__(self)

    def op_mul(self, other):
        return NotImplemented

    def __mul__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return self.op_mul(other)

    def __rmul__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return other.__mul__(self)

    def op_truediv(self, other):
        return NotImplemented

    def __truediv__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return self.op_truediv(other)

    def __rtruediv__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return other.__truediv__(self)

    def op_mod(self, other):
        return NotImplemented

    def __mod__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return self.op_mod(other)

    def __rmod__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return other.__mod__(self)

    def op_cmp(self, other):
        return NotImplemented

    def __cmp__(self, other):
        try:
            other = self.coerce(other)
        except CoercionError:
            return NotImplemented
        return self.op_cmp(other)
