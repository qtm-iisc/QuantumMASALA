"""Implements QuantumMASALA's primary data containers for storing
crystal-periodic quantities such as charge density, potentials, periodic part
of Bloch Wavefunction, etc.

The containers come in pairs, one for representing quantities in
real-space, and the other in G-space. The pair contains methods (``to_r()``/
``to_g()``) to generate its corresponding dual instance using the Fourier
transforms defined in the `qtm.gspace.GSpaceBase` class it contains.

There are 3 pairs of container classes in total:

1. `qtm.containers.FieldG`, `qtm.containers.FieldR`: for field quantitites
   such as charge density and its gradients, potentials, etc.
2. `qtm.containers.WavefunG`, `qtm.containers.WavefunR`: for the periodic
   part of Bloch Wavefunctions.
3. `qtm.containers.WavefunSpinG`, `qtm.containers.WavefunSpinR`: similar to the
   previous pair, but includes spin for noncollinear calculations.

The above classes subclass the `qtm.containers.Buffer` class, which implements
support for NumPy's Universal Functions using the
`numpy.lib.mixins.NDArrayOperatorsMixin` class. This enables operations such
as ``+``, ``-``, ``*``, ``/``, ``>=``, ``==``, etc. between
compatible instances. The `qtm.containers.Buffer` class supports:

1. all scalar `numpy.ufunc` operations where the operator is applied
   element-wise across all input arrays and returns a single value output(s)
   forming the output array
2. all `numpy.ufunc.reduce` operations, where the output will be cast back
   to the corresponding `qtm.containers.Buffer` instance only if the last axis
   is **NOT** reduced, else an array is returned.
3. The ``out`` keyword argument in `numpy.ufunc`, allowing for in-place
   operations such as ``+=``, ``-=``, ``*=``, etc.

The ``qtm.containers.Buffer`` class also supports array indexing, slicing and
list-comprehension like ``numpy.ndarray`` but with a key-difference; the last
axis, which represents the basis of the space cannot be indexed/sliced. It is
instead accessed by getting the underlying array through
`qtm.containers.Buffer.data` attribute.

Finally, analogous to array creation routines in NumPy, the
`qtm.containers.Buffer` class implement creation routines such as:

* `qtm.containers.Buffer.empty`
* `qtm.containers.Buffer.zeros`
* `qtm.containers.Buffer.from_array`
* `qtm.containers.Buffer.copy`

Refer to the documentation of the subclasses for details on additional methods
implemented.
"""
from .buffer import *
from .field import *
from .wavefun import *
