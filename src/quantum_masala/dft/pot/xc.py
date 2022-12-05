import numpy as np

from pylibxc import LibXCFunctional
import pylibxc.flags as xc_flags

from quantum_masala.core import GField, rho_check, deloper
from quantum_masala import config


def _get_sigma(rhoaux: GField) -> np.ndarray:
    r"""Generates the contracted gradient 'sigma' of input electron density.
    Required by libxc for computing potentials for GGA Functionals.

    Parameters
    ----------
    rhoaux : GField
        Input Electron Density

    Returns
    -------
    sigma_r : np.ndarray
        'sigma' array to be passed to `pylibxc.LibXCFunctional` for evaluating
        GGA Functional

    Notes
    -----
    Refer to the `libxc manual`_ for further details

    .. _libxc manual: https://www.tddft.org/programs/libxc/manual/libxc-5.1.x/
    """
    grho = rhoaux.gspc
    numspin = rhoaux.shape[0]
    grad_rhoaux = deloper.compute_grad(rhoaux)
    grad_rhoaux_r = grad_rhoaux.r

    sigma_r = np.empty((2*numspin - 1, *grho.grid_shape), dtype='f8')
    sigma_r[0] = np.sum(grad_rhoaux_r[0] * grad_rhoaux_r[0], axis=0).real
    if numspin == 2:
        sigma_r[1] = np.sum(grad_rhoaux_r[0]*grad_rhoaux_r[1], axis=0).real
        sigma_r[2] = np.sum(grad_rhoaux_r[1]*grad_rhoaux_r[1], axis=0).real

    return np.copy(sigma_r.reshape(sigma_r.shape[0], -1).T, order='C')


def xc_compute(rho: GField, rhocore: GField,
               exch_name: str, corr_name: str) -> tuple[GField, float]:
    """Computes the XC Potential generated by input charge density.

    The XC functional is specified by ``exch_name`` and ``corr_name``. The core
    charge density ``rhocore`` is required for Non-local Core correction.

    Parameters
    ----------
    rho : GField
        Input electron density
    rhocore : GField
        Core electron density. Generated from pseudopotential data
    exch_name : str
        Name of Exchange Functional
    corr_name : str
        Name of Correlation Functional

    Returns
    -------
    v_xc : GField
        XC Potential
    en_xc : float
        Contribution of XC Potential to total energy (per unit cell)
    """
    rho_check(rho)
    rho_check(rhocore)
    if rho.gspc != rhocore.gspc:
        raise ValueError("'gspc' between 'rho' and 'rho_core' must be equal.")

    grho = rho.gspc
    numspin = rho.shape[0]
    rhoaux = rho + rhocore

    xc_spin = "unpolarized" if numspin == 1 else "polarized"
    exch_func = LibXCFunctional(exch_name, xc_spin)
    corr_func = LibXCFunctional(corr_name, xc_spin)

    need_grad = sum(
        True if xcfunc.get_family() in
        [xc_flags.XC_FAMILY_GGA, xc_flags.XC_FAMILY_HYB_GGA]
        else False
        for xcfunc in [exch_func, corr_func]
    )

    for xcfunc in [exch_func, corr_func]:
        if xcfunc.get_family() == xc_flags.XC_FAMILY_LDA:
            xcfunc.set_dens_threshold(config.libxc_thr_lda_rho)
        elif xcfunc.get_family() == xc_flags.XC_FAMILY_GGA:
            xcfunc.set_dens_threshold(config.libxc_thr_gga_rho)
            xcfunc.set_sigma_threshold(config.libxc_thr_gga_sig)

    xc_inp = {"rho": np.copy(np.transpose(rho.r.real.reshape(numspin, -1)), "C")}
    if need_grad:
        xc_inp["sigma"] = _get_sigma(rhoaux)
    v_xc = GField.zeros(grho, numspin)
    en_xc = 0

    for xcfunc in [exch_func, corr_func]:
        xcfunc_out = xcfunc.compute(xc_inp)
        zk_r = GField.from_array(grho, xcfunc_out['zk'].reshape(grho.grid_shape))
        v_r = np.reshape(xcfunc_out['vrho'].T, (numspin, *grho.grid_shape))
        v_xc += GField.from_array(grho, v_r)
        en_xc += np.sum(rho.integrate_r(zk_r)).real

        if need_grad:
            grad_rho = deloper.compute_grad(rho)
            grad_rho_r = grad_rho.r
            vsig_r = np.reshape(xcfunc_out['vsigma'].T, (-1, *grho.grid_shape))
            h_r = np.empty((numspin, 3, *grho.grid_shape), dtype='c16')
            if numspin == 1:
                h_r[0] = 2*vsig_r[0]*grad_rho_r[0]
            else:
                h_r[0] = 2*vsig_r[0]*grad_rho_r[0] + vsig_r[1]*grad_rho_r[1]
                h_r[1] = 2*vsig_r[2]*grad_rho_r[1] + vsig_r[1]*grad_rho_r[0]
            h = GField.from_array(grho, h_r)
            div_h = deloper.compute_div(h)

            v_xc -= div_h

    return v_xc, en_xc.real
