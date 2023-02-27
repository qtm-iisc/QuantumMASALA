__all__ = ["rho_check", "rho_normalize", "rho_atomic"]

from typing import Optional
from numbers import Real

import numpy as np

from quantum_masala.core import Crystal, GField, GSpace
from quantum_masala.pseudo.loc import loc_generate_rhoatomic
from quantum_masala import pw_logger

EPS5 = 1E-4
EPS10 = 1E-10


def rho_check(rho: GField, gspc_rho: Optional[GSpace] = None,
              is_spin: Optional[bool] = None) -> None:
    """Checks if input ``rho`` matches the expected type and shape for
    an electron density object.

    Parameters
    ----------
    rho
        Represents the charge density of system
    gspc_rho
        GSp
    is_spin
        Spin-polarized if True

    Raises
    -------
    TypeError
        If either ``rho`` is not an instance of ``GField`` or ``is_spin`` is
        neither None nor boolean
    ValueError
        If ``rho.shape`` is not ``(1, )`` nor ``(2, )``.
    """
    # Validate the types of inputs
    if not isinstance(rho, GField):
        raise TypeError("'rho' must be an instance of 'GField'. "
                        f"got {type(rho)}.")
    if is_spin is not None and not isinstance(is_spin, bool):
        raise TypeError("'is_spin' must be either None or a boolean. "
                        f"got {type(is_spin)}.")

    # Check shape of 'rho'
    if is_spin is None:
        if rho.shape not in [(1, ), (2, )]:
            raise ValueError("'rho.shape' must be (1,) or  (2,). "
                             f"got {rho.shape}")
    elif rho.shape != (1 + is_spin, ):
        raise ValueError(f"'rho.shape' must be ({1 + is_spin}, ). "
                         f"got {rho.shape}")

    if gspc_rho is None:
        return
    if rho.gspc != gspc_rho:
        raise ValueError("'rho.gspc' does not match 'gspc_rho'.")


def rho_normalize(rho: GField, numel: float) -> GField:
    """Normalizes the electron density represented by ``rho`` to match the
    number of electrons per unit cell ``numel``

    Parameters
    ----------
    rho
        Input electron density
    numel
        Number of electrons per unit cell. Must be positive.

    Returns
    -------
    rho
        Input electron density that is normalized to contain ``numel`` number
        of electrons per unit cell

    Raises
    ------
    ValueError
        If ``numel`` is not positive or net charge in unit cell given by ``rho``
        is too small
    """
    if numel < EPS5:
        raise ValueError("'numel' must be positive. got:"
                         f"numel = {numel}")

    rho_check(rho)
    grho = rho.gspc
    rho_r = rho.to_rfield()

    idx_neg_r = np.nonzero(rho_r.r.real < 0)
    if len(idx_neg_r[0]) != 0:
        del_rho = -np.sum(rho_r.r[idx_neg_r]) * grho.reallat_dv
        pw_logger.warn("negative values found in `rho.r`.\n"
                       f"error: {del_rho}.")
        # rho_r.r[:] = np.abs(rho_r.r)

    rho_int = rho_r.integrate(axis=0)

    if rho_int < EPS10:
        raise ValueError("values in 'rho.r' too small to normalize.\n"
                         f"computed total charge = {rho_int}")
    if np.abs(rho_int - numel) > EPS5:
        pw_logger.warn(f"total charge renormalized from {rho_int} to {numel}")
    fac = numel / rho_int
    rho_r *= fac

    rho = rho_r.to_gfield()
    rho.Bcast()
    return rho


def rho_atomic(crystal: Crystal, grho: GSpace, is_spin: bool,
               l_mag: Optional[list[float]] = None) -> GField:
    """Constructs rho from atomic charge superposition with option to set
    magnetization for each atomic species in crystal

    Parameters
    ----------
    crystal
        Descrption of the Crystal
    grho
        Desciption of the PW Basis
    is_spin
        Generates spin-dependent density if True, else spin-unpolarized
    l_mag
        magnetization for each species in ``crystal.l_atoms``. Mandatory if
        ``is_spin`` is True, else ignored. Values must lie between
        +1 (all spin-up) and -1 (all spin-down)
    Returns
    -------
    rho_atomic
        Rho from superposition of Atomic Densities
    """
    l_atoms = crystal.l_atoms

    # Validating values in 'l_mag'
    if is_spin:
        if l_mag is None:
            raise ValueError("'l_mag' cannot be None if `'is_spin' is True.")
        if len(l_mag) != len(l_atoms) or not all(isinstance(x, Real) for x in l_mag):
            raise ValueError(
                "'l_mag' must be a list of 'len(crystal.l_atoms)' real numbers. got:\n"
                f"len(crystal.l_atoms) = {len(l_atoms)}\n"
                f"l_mag = {l_mag}"
            )

    if is_spin:
        rho = GField.zeros(grho, 2)
        for typ, mag in zip(l_atoms, l_mag):
            if abs(mag) > 1:
                mag = 1 if mag > 1 else -1  # Clipping values
            rho += [1 + mag/2, 1 - mag/2] * loc_generate_rhoatomic(typ, grho)
    else:
        rho = GField.zeros(grho, 1)
        for typ in l_atoms:
            rho += loc_generate_rhoatomic(typ, grho)

    return rho
