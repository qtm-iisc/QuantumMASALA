"""QuantumMASALA's primary data container for storing periodic functions/fields

Implements ``RField`` and ``GField``; array-like containers that represent
multidimensional periodic functions in real-space and G-space basis respectively.
The containers can behave like arrays with implemented binary operations, and
provides methods to convert between the two basis with a simple function call.


"""
from .buffer import *
from .field import *
from .wavefun import *
