# from future import __annotations__
__all__ = ["RotateWfn"]

import numpy as np

from qtm.gspace import GkSpace
from qtm.gspace.base import cryst2idxgrid
from qtm.crystal import Crystal
from qtm.containers import WavefunGType, get_WavefunG

from qtm.config import MPI4PY_INSTALLED

from qtm.msg_format import *
from qtm.constants import TPIJ

ROUND_PREC: int = 6


class RotateWfn:
    r"""Rotates a Bloch wavefunction from one k-point to a symmetry-related
    k-point in the Brillouin Zone.

    Given a symmetry operation :math:`\{S | \boldsymbol\tau\}` of
    ``crystal`` (real-space rotation `S` and fractional translation
    :math:`\boldsymbol\tau`, both in crystal coordinates), the periodic
    part of the Bloch wavefunction transforms as

    .. math::
        u_{\mathbf{k}'}(\mathbf{r}) = e^{-i(\mathbf{k}'+\mathbf{G}_0)\cdot
        \boldsymbol\tau} u_{\mathbf{k}}\big(S^{-1}(\mathbf{r}-
        \boldsymbol\tau)\big),

    where :math:`\mathbf{k}' = S\mathbf{k} \pmod{\mathbf{G}_0}` (with an
    extra minus sign on `S` and a complex conjugation if the operation is
    combined with time reversal). In terms of the plane-wave coefficients
    of `gkspc_src`/`gkspc_dest`, this amounts to

    .. math::
        c_{\mathbf{k}', \mathbf{G}'} = c_{\mathbf{k}, \mathbf{G}}\,
        e^{-i(\mathbf{k}'+\mathbf{G}')\cdot\boldsymbol\tau}, \quad
        \mathbf{G}' = S\mathbf{G} + \mathbf{G}_0,

    which is what `rotate` evaluates, after working out the relating
    symmetry operation `isymm` (and whether it requires time reversal) and
    the corresponding map between the G-vectors of `gkspc_src` and
    `gkspc_dest` once in the constructor.

    Parameters
    ----------
    crystal : Crystal
        Crystal whose space-group symmetries relate the two k-points.
    gkspc_src : GkSpace
        G-space of the wavefunction to be rotated, centered at k-point
        :math:`\mathbf{k}`.
    gkspc_dest : GkSpace
        G-space of the rotated wavefunction, centered at k-point
        :math:`\mathbf{k}'`. Must share the same parent `GSpace` (i.e.
        charge-density grid) as `gkspc_src`.
    isymm : int, optional
        Index into ``crystal.symm`` of the symmetry operation relating the
        two k-points. If not given, all operations are searched (see
        ``crystal.symm.find_symm``) and the *first* one found is used --
        since a k-point pair can be related by more than one symmetry
        operation (e.g. when `gkspc_src`'s little group is nontrivial),
        this arbitrary choice may not be the operation you want. Call
        ``crystal.symm.find_symm`` beforehand to list every candidate and
        pass the one you want explicitly.
    time_reversal : bool, optional
        Whether the symmetry operation `isymm` must be combined with time
        reversal i.e. :math:`\mathbf{k}' = -S\mathbf{k} \pmod{\mathbf{G}_0}`
        and :math:`c_{\mathbf{k}',\mathbf{G}'} = c_{\mathbf{k},\mathbf{G}}^*
        e^{-i(\mathbf{k}'+\mathbf{G}')\cdot\boldsymbol\tau}`. If `None`,
        both possibilities are tried while searching for `isymm`.
    """

    def __init__(
        self,
        crystal: Crystal,
        gkspc_src: GkSpace,
        gkspc_dest: GkSpace,
        isymm: int = None,
        time_reversal: bool = None,
    ):
        if not isinstance(crystal, Crystal):
            raise TypeError(type_mismatch_msg("crystal", crystal, Crystal))
        if not isinstance(gkspc_src, GkSpace):
            raise TypeError(type_mismatch_msg("gkspc_src", gkspc_src, GkSpace))
        if not isinstance(gkspc_dest, GkSpace):
            raise TypeError(type_mismatch_msg("gkspc_dest", gkspc_dest, GkSpace))

        if MPI4PY_INSTALLED:
            from qtm.mpi.gspace import DistGkSpace

            if isinstance(gkspc_src, DistGkSpace):
                gkspc_src = gkspc_src.gkspc_glob
            if isinstance(gkspc_dest, DistGkSpace):
                gkspc_dest = gkspc_dest.gkspc_glob

        if gkspc_src.gwfn is not gkspc_dest.gwfn:
            raise ValueError(
                obj_mismatch_msg(
                    "gkspc_src.gwfn",
                    gkspc_src.gwfn,
                    "gkspc_dest.gwfn",
                    gkspc_dest.gwfn,
                )
            )

        self.gkspc_src = gkspc_src
        self.gkspc_dest = gkspc_dest

        # Symmetry operations act on k-points/G-vectors in crystal
        # coordinates; kept on the host regardless of the G-space's array
        # backend since they are tiny and only used to locate 'isymm'.
        recilat_rot = np.asarray(crystal.symm.recilat_rot)
        reallat_trans = np.asarray(crystal.symm.reallat_trans)

        k_src = np.around(np.asarray(gkspc_src.k_cryst, dtype="f8"), ROUND_PREC)
        k_dest = np.around(np.asarray(gkspc_dest.k_cryst, dtype="f8"), ROUND_PREC)

        if isymm is None:
            matches = crystal.symm.find_symm(k_src, k_dest, time_reversal)
            if not matches:
                raise ValueError(
                    f"no symmetry operation of 'crystal.symm' relates "
                    f"k_src={tuple(k_src)} to k_dest={tuple(k_dest)}"
                    + (
                        " (with or without time reversal)"
                        if time_reversal is None
                        else " combined with time reversal"
                        if time_reversal
                        else ""
                    )
                )
            isymm, time_reversal = matches[0]
        elif time_reversal is None:
            time_reversal = False

        self.isymm = isymm
        self.time_reversal = time_reversal

        sign = -1 if time_reversal else 1
        S = sign * recilat_rot[isymm]
        tau = reallat_trans[isymm]

        # k' = S@k (mod G0); G0 is the reciprocal lattice vector needed to
        # bring 'S@k_src' back to the 'gkspc_dest.k_cryst' label
        k_rot = S @ k_src
        g0 = np.rint(k_rot - k_dest)
        if not np.allclose(k_rot - g0, k_dest, atol=10.0 ** (-ROUND_PREC)):
            raise ValueError(
                f"'gkspc_dest.k_cryst' = {gkspc_dest.k_cryst} is not related to "
                f"'gkspc_src.k_cryst' = {gkspc_src.k_cryst} by symmetry operation "
                f"#{isymm} of 'crystal.symm'"
                + (" combined with time reversal" if time_reversal else "")
            )

        # G' = S@G + G0: G-vectors of 'gkspc_src', rotated and shifted by
        # the umklapp vector so that they are labelled consistently with
        # 'gkspc_dest.g_cryst'
        g_cryst_src = gkspc_src.g_cryst
        S = np.asarray(S, like=g_cryst_src)
        g0 = np.asarray(np.rint(g0).astype("i8"), like=g_cryst_src)
        g_cryst_rot = np.tensordot(S, g_cryst_src, axes=1) + g0[:, np.newaxis]

        # Locating the position of each rotated G-vector within
        # 'gkspc_dest.g_cryst' (which, being a 'GSpaceBase', is already
        # sorted/indexed by its FFT-grid position, i.e. 'idxgrid')
        idxgrid_rot = cryst2idxgrid(gkspc_dest.grid_shape, g_cryst_rot)
        idxsort_dest = np.argsort(gkspc_dest.idxgrid)
        idxgrid_dest_sorted = gkspc_dest.idxgrid[idxsort_dest]
        pos = np.searchsorted(idxgrid_dest_sorted, idxgrid_rot)

        matched = np.zeros(pos.shape, dtype=bool)
        in_bounds = pos < idxgrid_dest_sorted.shape[0]
        matched[in_bounds] = (
            idxgrid_dest_sorted[pos[in_bounds]] == idxgrid_rot[in_bounds]
        )
        if gkspc_src.size_g != gkspc_dest.size_g or not bool(np.all(matched)):
            raise ValueError(
                "failed to map the G-vectors of 'gkspc_src', rotated by symmetry "
                f"operation #{isymm}, onto the G-vectors of 'gkspc_dest'. Make "
                "sure both are constructed from the same 'gwfn' using the same "
                "'ecutwfn'."
            )
        self.idx_dest = idxsort_dest[pos]

        # Phase e^{-i(k'+G').tau} picked up from the fractional translation
        # of the symmetry operation, indexed by the G-vectors of 'gkspc_src'
        tau = np.asarray(tau, like=g_cryst_src)
        k_dest_g = np.asarray(k_dest, like=g_cryst_src)
        kg_dest = k_dest_g[:, np.newaxis] + g_cryst_rot
        self.phase = np.exp(-TPIJ * np.sum(tau[:, np.newaxis] * kg_dest, axis=0))

    def rotate(self, wfn_src: WavefunGType) -> WavefunGType:
        """Rotates a wavefunction sampled on `gkspc_src` to `gkspc_dest`.

        Parameters
        ----------
        wfn_src : WavefunGType
            Wavefunction(s) to rotate; its `gkspc` must be `gkspc_src` (or,
            with ``mpi4py`` installed, a distributed instance thereof, in
            which case it is gathered onto every process first).

        Returns
        -------
        WavefunGType
            Rotated wavefunction(s), sampled on `gkspc_dest`, with the same
            `shape` and `numspin` as `wfn_src`.
        """
        if not isinstance(wfn_src, WavefunGType):
            raise TypeError(type_mismatch_msg("wfn_src", wfn_src, WavefunGType))

        is_dist = False
        if MPI4PY_INSTALLED:
            from qtm.mpi.containers import DistBufferType

            if isinstance(wfn_src, DistBufferType):
                is_dist = True
        wfn_src_ = wfn_src.allgather() if is_dist else wfn_src

        if wfn_src_.gkspc is not self.gkspc_src:
            raise ValueError(
                obj_mismatch_msg(
                    "wfn_src.gkspc", wfn_src_.gkspc, "self.gkspc_src", self.gkspc_src
                )
            )

        shape, numspin = wfn_src_.shape, wfn_src_.numspin
        wfn_dest = get_WavefunG(self.gkspc_dest, numspin).empty(shape)

        data_src = wfn_src_.data.reshape((*shape, numspin, self.gkspc_src.size_g))
        data_dest = wfn_dest.data.reshape((*shape, numspin, self.gkspc_dest.size_g))

        data_dest[..., self.idx_dest] = (
            np.conj(data_src) if self.time_reversal else data_src
        ) * self.phase

        return wfn_dest
