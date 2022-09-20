

## Convention


### Variable names
Refer to the `Naming Convention` section in [PEP-8](https://peps.python.org/pep-0008/#naming-conventions)
#### Prefixes:
+ `numx` : Number of quantity represented by `x`
+ `l_x` : List of quantities of type `x`
+ `idxa` : Index/Indices labelled by `a`
+ `ia` : Same as above (mainly used for local variables)
#### Suffixes
+ `_cart` : Arrays containing vector components in cartesian basis
+ `_cryst` : Arrays containing vector components in crystal basis
+ `_alat` : Arrays containing vectors in **Real Space** in units of lattice parameter a
+ `_tpiba` : Arrays containing vectors in **Reciprocal Space** in units of lattice parameter b = $2\pi/a$

### List of vectors as array
Unlike most codes that are written in languages with Row-major ordering,
list of vectors in an array are in a column-major format. That is,
a list containing ten 3D vectors will be stored in an array of shape `(3, 10)`
instead of `(10, 3)`. This aligns with mathematical representation of vectors as Columns.

The actual ordering of data in memory is 'inconsequential' as it is taken care
of by `NumPy` where the last axis is the fastest index.



