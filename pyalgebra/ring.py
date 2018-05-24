from abc import ABC, abstractmethod, abstractproperty

class Ring(ABC):
    @abstractproperty
    def element_class(self):
        """
        Return the class of an element of this ring
        """
        pass

    @abstractmethod
    def coerce(self, elt):
        """
        Try to coerce `elt` to be a ring element.
        """
        pass

    @abstractmethod
    def zero(self):
        """
        Return the zero element of the ring
        """
        pass

    @abstractmethod
    def one(self):
        """
        Return the one element of the ring
        """
        pass

    @abstractmethod
    def an_element(self):
        """
        Return an element of the ring
        """
        pass

    @abstractmethod
    def random_element(self):
        """
        Return a random element of the ring.  The distribution depends on the
        ring.
        """
        pass

    @abstractmethod
    def order(self):
        """
        Return the order of the ring
        """
        pass
