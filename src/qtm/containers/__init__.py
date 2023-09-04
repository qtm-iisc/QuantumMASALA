"""Implements QuantumMASALA's primary data containers for storing
crystal-periodic quantities such as charge density, potentials, periodic part
of Bloch Wavefunction, etc.

The containers come in pairs, one for representing quantities in
real-space, and the other in G-space. The pair contains methods (``to_r()``/
``to_g()``) to generate its corresponding dual instance using the Fourier
transforms defined in the `qtm.gspace.GSpaceBase` class it contains.

There are 2 pairs of container classes in total:

1. `qtm.containers.FieldGType`, `qtm.containers.FieldRType`: for field quantitites
   such as charge density and its gradients, potentials, etc.
2. `qtm.containers.WavefunGType`, `qtm.containers.WavefunRType`: for the periodic
   part of Bloch Wavefunctions.

The above classes require a `GSpace` instance (for `FieldGType`/`FieldRType`)
or a `GkSpace` instance (for `WavefunGType`/`WavefunRType`) for its
definition. The following routines return this implementation:

* `qtm.containers.get_FieldG`: returns `FieldGType` impl named `FieldG`
* `qtm.containers.get_FieldR`: returns `FieldRType` impl named `FieldR`
* `qtm.containers.get_WavefunG`: returns `WavefunGType` impl named `WavefunG`
* `qtm.containers.get_WavefunR`: returns `WavefunRType` impl named `WavefunR`

Note that the functions are cached i.e, they return the same object, which
in this case is a class itself, for the same input (a `qtm.gspace.GSpaceBase`
instance). This allows checking if two containers are compatible i.e. have the
same basis by simply checking if their types match like so:
``type(buf1) is type(buf2)``.

The above classes subclass the `qtm.containers.BufferType` class, which
implements support for NumPy's Universal Functions using the
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

The `qtm.containers.BufferType` class also supports array indexing, slicing and
list-comprehension like `numpy.ndarray` but with a key-difference; the last
axis, which represents the basis of the space cannot be indexed/sliced. It is
instead accessed by getting the underlying array through
`qtm.containers.BufferType.data` attribute.

Finally, analogous to array creation routines in NumPy, the
`qtm.containers.Buffer` class implement creation routines such as:

* `qtm.containers.BufferType.empty`
* `qtm.containers.BufferType.zeros`
* `qtm.containers.BufferType.from_array`
* `qtm.containers.BufferType.copy`

Refer to the documentation of the subclasses for details on additional methods
implemented.
"""
from .buffer import *
from .field import *
from .wavefun import *
