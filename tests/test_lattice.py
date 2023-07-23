import pytest

import numpy as np
from qtm.lattice import Lattice

ALAT = 2.0


@pytest.fixture
def sc_lattice():
    return Lattice(ALAT * np.eye(3))


@pytest.fixture
def fcc_lattice():
    primvec = (ALAT/2) * np.array([
        [-1, 0, 1],
        [ 0, 1, 1],
        [-1, 1, 0]
    ]).T
    return Lattice(primvec)


@pytest.fixture
def bcc_lattice():
    primvec = (ALAT/2) * np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [-1, -1,  1]
    ]).T
    return Lattice(primvec)


def test_cart2cryst():
    lattice = fcc_lattice()