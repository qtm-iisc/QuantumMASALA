from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


class Exchange:
    """Exact exchange (Fock) operator with optional ACE compression.

    The exchange operator is built from the occupied wavefunctions. During HF
    SCF, the inner loop keeps the exchange operator fixed, while the outer EXX
    loop updates the occupied wavefunctions and rebuilds the exchange operator
    and, when enabled, the ACE projectors.

    Parameters
    ----------
    gkspc : GkSpace
        Basis / FFT grids.
    psi : WavefunG-like container
        Wavefunctions used to initialize the occupied set.
    occ : array-like
        Occupations (including spin factor if desired).
    volume : float
        Cell volume.
    method : str
        Coulomb kernel method: "none" or "vcut_spherical".
    Rc : float
        Radius for spherical truncation (required if method="vcut_spherical").
    use_ace : bool
        If True, build and use ACE projectors for fast application in the inner
        SCF loop.
    ace_jitter : float
        Relative stabilization used when factorizing the ACE overlap matrix.
    """

    def __init__(
        self,
        gkspc,
        psi,
        occ,
        volume,
        method="none",
        Rc=None,
        *,
        use_ace=True,
        ace_jitter=1e-12,
    ):
        self.gkspc = gkspc
        self.psi = psi
        self.occ = np.asarray(occ, dtype=float)
        self.volume = float(volume)

        # Coulomb kernels acting on periodic fields use |G|^2.
        # Using |G+k|^2 is equivalent only at Gamma.
        if hasattr(gkspc, "g_cryst"):
            G2 = gkspc.recilat.norm2(gkspc.g_cryst, "cryst")
        else:
            # Gamma-point fallback
            G2 = gkspc.gk_norm2
        G = np.sqrt(G2)

        self.kernel = np.zeros_like(G2, dtype=float)

        # ======================================================
        # 1) Bare periodic Coulomb
        # ======================================================
        if method == "none":
            mask = G2 > 0.0
            self.kernel[mask] = 4.0 * np.pi / (G2[mask] * self.volume)

        # ======================================================
        # 2) Spherical truncation
        # ======================================================
        elif method == "vcut_spherical":
            if Rc is None:
                raise ValueError("Rc must be provided for vcut_spherical")

            Rc = float(Rc)
            mask = G2 > 1e-14
            self.kernel[mask] = (
                4.0 * np.pi / (G2[mask] * self.volume) * (1.0 - np.cos(G[mask] * Rc))
            )

            # analytic G -> 0 limit
            self.kernel[~mask] = 2.0 * np.pi * Rc**2 / self.volume

        else:
            raise ValueError(f"Unknown exchange kernel method: {method}")

        # ------------------------------------------------------
        # ACE controls
        # ------------------------------------------------------
        self.use_ace = bool(use_ace)
        self.ace_jitter = float(ace_jitter)

        # ------------------------------------------------------
        # Cached occupied wavefunctions
        # ------------------------------------------------------
        self._psi_occ_g = None
        self._psi_occ_r = None
        self._occ_idx = None
        self._occ = None
        self._scale = None

        # ------------------------------------------------------
        # ACE projectors (xi) in G-space
        # ------------------------------------------------------
        self._xi_g = None

        # Build initial caches (and ACE if enabled)
        self.update_orbitals(self.psi, self.occ, build_ace=True)

    # ---------------------------------------------------------------------
    # Update occupied wavefunctions
    # ---------------------------------------------------------------------
    def update_orbitals(self, psi, occ, *, build_ace=True):
        """Update the occupied wavefunctions used to build the exchange operator."""
        self.psi = psi
        self.occ = np.asarray(occ, dtype=float)

        occ_idx = np.where(self.occ != 0)[0]
        self._occ_idx = occ_idx
        self._occ = self.occ[occ_idx]

        if len(occ_idx) == 0:
            self._psi_occ_g = []
            self._psi_occ_r = []
            self._scale = None
            self._xi_g = None
            return

        # Compensate unnormalized FFT convention
        ngrid = psi[int(occ_idx[0])].to_r().data.size
        self._scale = 1.0 / np.sqrt(ngrid)

        # Occupied wavefunctions in G-space
        self._psi_occ_g = [psi[int(i)].copy() for i in occ_idx]

        # Cache scaled real-space versions (used by full exchange application)
        psi_occ_r = []
        for psi_g in self._psi_occ_g:
            w_r = psi_g.to_r()
            w_r *= self._scale
            psi_occ_r.append(w_r)
        self._psi_occ_r = psi_occ_r

        # Rebuild ACE projectors (optional)
        self._xi_g = None
        if self.use_ace and build_ace:
            self.build_ace()

    # ---------------------------------------------------------------------
    # Local container-safe real-space algebra helpers
    # ---------------------------------------------------------------------
    def _mul_wr(self, a_r, b_r):
        """Pointwise product of two WavefunR-like objects."""
        out = a_r.copy()
        out.data[:] = a_r.data * b_r.data
        return out

    def _conj_mul_wr(self, a_r, b_r):
        """Pointwise product conj(a_r) * b_r for two WavefunR-like objects."""
        out = b_r.copy()
        out.data[:] = np.conjugate(a_r.data) * b_r.data
        return out

    # ---------------------------------------------------------------------
    # Full (uncompressed) exchange application
    # ---------------------------------------------------------------------
    def _apply_fock_full(self, psi_in):
        """Return Vx|psi_in> using the full occupied set."""
        if self._psi_occ_r is None or len(self._psi_occ_r) == 0:
            out = psi_in.copy()
            out.data[:] = 0.0
            return out

        psi_in_r = psi_in.to_r()
        psi_in_r *= self._scale

        vx_psi_r = psi_in_r.copy()
        vx_psi_r.data[:] = 0.0

        for f_m, psi_m_r in zip(self._occ, self._psi_occ_r):
            if f_m == 0:
                continue

            rho_r = self._conj_mul_wr(psi_m_r, psi_in_r)
            rho_G = rho_r.to_g()

            np.multiply(rho_G.data, self.kernel, out=rho_G.data)

            v_mn_r = rho_G.to_r()
            v_mn_r *= self._scale

            prod_r = self._mul_wr(psi_m_r, v_mn_r)
            vx_psi_r.data[:] += (-0.5 * float(f_m)) * prod_r.data

        return vx_psi_r.to_g()

    # ---------------------------------------------------------------------
    # ACE build + ACE application
    # ---------------------------------------------------------------------
    def build_ace(self):
        """Build ACE projectors xi_k in G-space.

        The construction cost is one full application of Vx to each occupied
        wavefunction (phi_i). Once built, applying Vx in the inner SCF
        loop becomes a low-rank projector:

            Vx^ACE |psi> = - sum_k |xi_k> <xi_k|psi>

        (sign consistent with the 2016 Lin Lin ACE formulation).
        """
        nocc = len(self._psi_occ_g) if self._psi_occ_g is not None else 0
        if nocc == 0:
            self._xi_g = None
            return

        # 1) W_i = Vx_full |phi_i>
        w_list = [self._apply_fock_full(phi_i) for phi_i in self._psi_occ_g]

        # 2) M_ij = <phi_i|W_j> = <phi_i|Vx|phi_j>
        M = np.empty((nocc, nocc), dtype=complex)
        for i, phi_i in enumerate(self._psi_occ_g):
            vi = phi_i.data.ravel()
            for j, w_j in enumerate(w_list):
                M[i, j] = np.vdot(vi, w_j.data.ravel())

        # Hermitize to reduce numerical noise
        M = 0.5 * (M + M.conj().T)

        # B = -M should be positive definite (Vx is negative semidefinite)
        B = -M
        B = 0.5 * (B + B.conj().T)

        # Stabilize / factorize
        diag_scale = float(np.max(np.abs(np.diag(B)))) if nocc > 0 else 1.0
        jitter = self.ace_jitter * max(diag_scale, 1.0)
        B_stab = B + jitter * np.eye(nocc, dtype=complex)

        xi_list = []

        try:
            # Cholesky on B = L L^H
            L = np.linalg.cholesky(B_stab)

            # C = L^{-H}, so that C C^H = B^{-1}
            C = np.linalg.solve(
                L.conj().T,
                np.eye(nocc, dtype=complex),
            )

            for k in range(nocc):
                xi_k = w_list[0].copy()
                xi_k.data[:] = 0.0
                for j in range(nocc):
                    xi_k.data[:] += w_list[j].data * C[j, k]
                xi_list.append(xi_k)

        except np.linalg.LinAlgError:
            # Fallback: eigen-based B^{-1/2}
            w, V = np.linalg.eigh(B_stab)
            w = np.maximum(w, 1e-14 * np.max(w))
            Binv_sqrt = (V * (w ** -0.5)) @ V.conj().T

            for k in range(nocc):
                xi_k = w_list[0].copy()
                xi_k.data[:] = 0.0
                for j in range(nocc):
                    xi_k.data[:] += w_list[j].data * Binv_sqrt[j, k]
                xi_list.append(xi_k)

        self._xi_g = xi_list

    def _apply_fock_ace(self, psi_in):
        """Return Vx^ACE|psi_in> using cached ACE projectors."""
        if self._xi_g is None or len(self._xi_g) == 0:
            # Fall back to full if ACE not available
            return self._apply_fock_full(psi_in)

        out = psi_in.copy()
        out.data[:] = 0.0

        vin = psi_in.data.ravel()
        for xi in self._xi_g:
            # <xi|psi>
            c = np.vdot(xi.data.ravel(), vin)
            out.data[:] += (-c) * xi.data

        return out

    # ---------------------------------------------------------------------
    # Public application API
    # ---------------------------------------------------------------------
    def apply_fock(self, psi_in):
        """Return Vx|psi_in>.

        If ACE is enabled and built, uses the ACE projector form for speed.
        """
        if self.use_ace and (self._xi_g is not None):
            return self._apply_fock_ace(psi_in)
        return self._apply_fock_full(psi_in)

    def h_psi(self, l_psi, l_hpsi):
        """Add exchange contribution to a batch: l_hpsi += Vx l_psi."""
        l_psi = l_psi.reshape(-1)
        l_hpsi = l_hpsi.reshape(-1)

        for psi, hpsi in zip(l_psi, l_hpsi):
            hpsi += self.apply_fock(psi)

    # ---------------------------------------------------------------------
    # Energies / diagnostics
    # ---------------------------------------------------------------------
    def fock_energy(self, psi, occ):
        """Return fock = sum_i f_i <psi_i|Vx|psi_i> for a given band set.

        This matches QE's notion of the "Fock energy" fock0/fock1/fock2, i.e.
        without the 1/2 factor used in the exchange energy.
        """
        occ = np.asarray(occ, dtype=float)
        idx = np.where(occ != 0.0)[0]
        fock = 0.0

        for i in idx:
            psi_i = psi[int(i)]
            vxpsi_i = self.apply_fock(psi_i)
            x_i = np.vdot(psi_i.data.ravel(), vxpsi_i.data.ravel()).real
            fock += float(occ[int(i)]) * float(x_i)

        return float(fock)

    def energy(self):
        """Return the exchange energy for the cached occupied wavefunctions."""
        Ex = 0.0

        if self._psi_occ_r is None or len(self._psi_occ_r) == 0:
            return 0.0

        for fi, psi_i_r in zip(self._occ, self._psi_occ_r):
            if fi == 0:
                continue

            for fj, psi_j_r in zip(self._occ, self._psi_occ_r):
                if fj == 0:
                    continue

                rho_r = self._conj_mul_wr(psi_j_r, psi_i_r)
                rho_G = rho_r.to_g()
                Y2 = np.abs(rho_G.data) ** 2

                Ex += float(fi) * float(fj) * np.sum(self.kernel * Y2)

        Ex *= -0.25
        return float(np.real(Ex))

# -----------------------------------------------------------------------------
# Multi-k periodic exact exchange
# -----------------------------------------------------------------------------
#
# For a target k-point, exact exchange includes occupied wavefunctions from all
# source k-points k' through q = k - k' and the Coulomb kernel v(|G + q|).
# MultiKExchangeDriver stores the occupied wavefunctions for all k-points, and
# MultiKExchangeBlock applies the exchange operator for one target k-point.
#
# K-points may be distributed across MPI k-groups. Occupied wavefunction data
# are gathered across k-group roots so that each target k-point includes the
# complete sum over source k-points. At present, HF/EXX supports k-point
# parallelism only: each k-group must contain one MPI rank. Distribution of the
# plane-wave G-space within a k-group is not yet supported.
#

@dataclass
class _OccState:
    """Occupied wavefunction at one k-point on the scaled real-space grid."""
    k_cryst: np.ndarray   # (3,) in crystal coordinates
    psi_r: object         # scaled WavefunR-like object (periodic part u_{mk'}(r))
    occ_wt: float         # (w_k' * f_occ_mk') with w_k' excluding spin degeneracy


class MultiKExchangeDriver:
    """Global EXX driver for a single spin channel.

    The driver stores occupied wavefunctions from all k-points and serves
    per-k exchange blocks attached to KSHam objects.

    Parameters
    ----------
    volume : float
        Real-space cell volume.
    nk_total : int
        Total number of k-points in the BZ sampling (used for minibz averaging).
    exxdiv_treatment : str
        "none", "vcut_spherical", or "minibz_sphere".
    exx_Rc : float | None
        Coulomb truncation radius if using vcut_spherical.
    occ_spin_factor : float
        Multiply KSWfn.occ by this to get f_occ used inside exchange (typically 2.0
        for non-spin-polarized, 1.0 for spin-polarized).
    kweight_bz_factor : float
        Multiply KSWfn.k_weight by this to obtain the *BZ* weight excluding spin
        degeneracy (typically 0.5 for non-spin-polarized because upstream code uses
        k_weight *= 2 to account for spin, and 1.0 otherwise).
    use_ace : bool
        Whether ACE compression is enabled in each per-k exchange block.
    ace_jitter : float
        ACE stabilization jitter.
    kroot_comm : QTMComm | None
        Communicator connecting the roots of the MPI k-groups. If provided,
        occupied wavefunction data are gathered across k-groups when the
        exchange operator is updated. If None, all k-points are local.
    """

    def __init__(
        self,
        *,
        volume: float,
        nk_total: int,
        exxdiv_treatment: str,
        exx_Rc: float | None,
        occ_spin_factor: float,
        kweight_bz_factor: float,
        use_ace: bool = True,
        ace_jitter: float = 1e-12,
        kroot_comm=None,
    ):
        self.volume = float(volume)
        self.nk_total = int(nk_total)
        self.exxdiv_treatment = str(exxdiv_treatment)
        self.exx_Rc = None if exx_Rc is None else float(exx_Rc)
        self.occ_spin_factor = float(occ_spin_factor)
        self.kweight_bz_factor = float(kweight_bz_factor)
        self.use_ace = bool(use_ace)
        self.ace_jitter = float(ace_jitter)
        self.kroot_comm = kroot_comm

        self._occ_states: List[_OccState] = []
        self._blocks: Dict[Tuple[float, float, float], "MultiKExchangeBlock"] = {}

    @staticmethod
    def _k_key(k_cryst: Tuple[float, float, float] | np.ndarray) -> Tuple[float, float, float]:
        k = np.asarray(k_cryst, dtype=float).ravel()
        return (float(k[0]), float(k[1]), float(k[2]))

    def block_for_kswfn(self, kswfn) -> "MultiKExchangeBlock":
        """Return (and lazily create) the exchange block for this k-point."""
        key = self._k_key(kswfn.k_cryst)
        blk = self._blocks.get(key, None)
        if blk is not None:
            return blk

        w_k_bz = self.kweight_bz_factor * float(kswfn.k_weight)
        f_occ = self.occ_spin_factor * np.asarray(kswfn.occ, dtype=float)

        blk = MultiKExchangeBlock(
            gkspc=kswfn.gkspc,
            psi=kswfn.evc_gk,
            occ=f_occ,
            volume=self.volume,
            k_cryst=np.asarray(kswfn.k_cryst, dtype=float),
            k_weight_bz=w_k_bz,
            nk_total=self.nk_total,
            exxdiv_treatment=self.exxdiv_treatment,
            exx_Rc=self.exx_Rc,
            use_ace=self.use_ace,
            ace_jitter=self.ace_jitter,
        )
        blk._occ_states = self._occ_states
        self._blocks[key] = blk
        return blk

    def update_from_kswfns(self, l_kswfn: List, *, build_ace: bool = True):
        """Update the occupied wavefunctions used by multi-k exact exchange.

        In an MPI k-point calculation, ``l_kswfn`` contains only the k-points
        assigned to the current k-group. Each k-group first constructs its
        local occupied wavefunction data and then exchanges these data with the other
        k-group roots.

        Only plain NumPy arrays and scalar metadata are communicated.
        """
        if len(l_kswfn) == 0:
            raise ValueError(
                "l_kswfn must contain at least one local k-point"
            )

        # All k-points share the same real-space FFT grid.
        template_r = l_kswfn[0].evc_gk[0].to_r()

        ngrid = template_r.data.size
        scale = 1.0 / np.sqrt(ngrid)

        local_payloads = []

        # ----------------------------------------------------
        # Build occupied wavefunctions belonging to this k-group
        # ----------------------------------------------------
        for kswfn in l_kswfn:
            w_k_bz = (
                self.kweight_bz_factor
                * float(kswfn.k_weight)
            )

            f_occ = (
                self.occ_spin_factor
                * np.asarray(kswfn.occ, dtype=float)
            )

            occ_idx = np.where(f_occ != 0.0)[0]

            if len(occ_idx) == 0:
                continue

            k_cryst = np.asarray(
                kswfn.k_cryst,
                dtype=float,
            )

            for ib in occ_idx:
                psi_r = kswfn.evc_gk[int(ib)].to_r()
                psi_r *= scale

                occ_wt = (
                    w_k_bz
                    * float(f_occ[int(ib)])
                )

                local_payloads.append(
                    (
                        np.array(
                            k_cryst,
                            dtype=float,
                            copy=True,
                        ),
                        float(occ_wt),
                        np.array(
                            psi_r.data,
                            copy=True,
                        ),
                    )
                )

        # ----------------------------------------------------
        # Collect occupied wavefunctions from every k-group
        # ----------------------------------------------------
        if (
            self.kroot_comm is not None
            and self.kroot_comm.size > 1
        ):
            payloads_by_kgrp = (
                self.kroot_comm.allgather(
                    local_payloads
                )
            )

            global_payloads = [
                payload
                for group_payloads in payloads_by_kgrp
                for payload in group_payloads
            ]

        else:
            global_payloads = local_payloads

        # ----------------------------------------------------
        # Reconstruct local WavefunR containers
        # ----------------------------------------------------
        occ_states: List[_OccState] = []

        for (
            k_cryst,
            occ_wt,
            psi_r_data,
        ) in global_payloads:

            psi_r = template_r.copy()
            psi_r.data[:] = psi_r_data

            occ_states.append(
                _OccState(
                    k_cryst=np.asarray(
                        k_cryst,
                        dtype=float,
                    ),
                    psi_r=psi_r,
                    occ_wt=float(occ_wt),
                )
            )

        self._occ_states = occ_states

        # ----------------------------------------------------
        # Update exchange blocks for local target k-points
        # ----------------------------------------------------
        for kswfn in l_kswfn:

            blk = self.block_for_kswfn(kswfn)

            blk._occ_states = self._occ_states

            f_occ = (
                self.occ_spin_factor
                * np.asarray(
                    kswfn.occ,
                    dtype=float,
                )
            )

            blk.update_orbitals(
                kswfn.evc_gk,
                f_occ,
                build_ace=build_ace,
            )


class MultiKExchangeBlock(Exchange):
    """Per-k exchange operator Vx^k that is coupled to all occupied k'."""

    def __init__(
        self,
        *,
        gkspc,
        psi,
        occ,
        volume: float,
        k_cryst: np.ndarray,
        k_weight_bz: float,
        nk_total: int,
        exxdiv_treatment: str,
        exx_Rc: float | None,
        use_ace: bool,
        ace_jitter: float,
    ):
        # Initialize directly because Exchange.__init__ builds a q=0 kernel.
        self.gkspc = gkspc
        self.psi = psi
        self.occ = np.asarray(occ, dtype=float)
        self.volume = float(volume)

        self.k_cryst = np.asarray(k_cryst, dtype=float).ravel()
        self.k_weight_bz = float(k_weight_bz)
        self.nk_total = int(nk_total)
        self.exxdiv_treatment = str(exxdiv_treatment)
        self.exx_Rc = None if exx_Rc is None else float(exx_Rc)

        self.use_ace = bool(use_ace)
        self.ace_jitter = float(ace_jitter)

        self._occ_states: List[_OccState] = []
        self._psi_occ_g = None
        self._psi_occ_r = None
        self._occ_idx = None
        self._occ = None
        self._scale = None
        self._xi_g = None

        self._kernel_cache: Dict[Tuple[float, float, float], np.ndarray] = {}

        # Build ACE after the driver supplies occupied wavefunctions from all k-points.
        self.update_orbitals(self.psi, self.occ, build_ace=False)

    def _kernel_key(self, q_cryst: np.ndarray) -> Tuple[float, float, float]:
        q = np.asarray(q_cryst, dtype=float).ravel()
        return (
            float(np.round(q[0], 12)),
            float(np.round(q[1], 12)),
            float(np.round(q[2], 12)),
        )

    def _minibz_sphere_kernel_g0(self) -> float:
        recvol = float(self.gkspc.recilat.cellvol)
        nq = float(self.nk_total)
        q_cut = (recvol / nq * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        avg_invq2 = 3.0 / (q_cut * q_cut)
        return (4.0 * np.pi * avg_invq2) / self.volume

    def _kernel_for_q(self, q_cryst: np.ndarray) -> np.ndarray:
        key = self._kernel_key(q_cryst)
        kern = self._kernel_cache.get(key, None)
        if kern is not None:
            return kern

        q = np.asarray(q_cryst, dtype=float).ravel()

        Gq_cryst = self.gkspc.g_cryst.astype("f8").copy()
        Gq_cryst[0, :] += q[0]
        Gq_cryst[1, :] += q[1]
        Gq_cryst[2, :] += q[2]

        Gq2 = self.gkspc.recilat.norm2(Gq_cryst, "cryst")
        Gq = np.sqrt(Gq2)

        kern = np.zeros_like(Gq2, dtype=float)

        if self.exxdiv_treatment in ("none", "minibz_sphere"):
            mask = Gq2 > 0.0
            kern[mask] = 4.0 * np.pi / (Gq2[mask] * self.volume)

            if self.exxdiv_treatment == "minibz_sphere":
                if np.allclose(q, 0.0, atol=1e-12):
                    kern[~mask] = self._minibz_sphere_kernel_g0()

        elif self.exxdiv_treatment == "vcut_spherical":
            if self.exx_Rc is None:
                raise ValueError("exx_Rc must be set for vcut_spherical")

            Rc = float(self.exx_Rc)
            mask = Gq2 > 1e-14
            kern[mask] = (
                4.0 * np.pi / (Gq2[mask] * self.volume) * (1.0 - np.cos(Gq[mask] * Rc))
            )
            kern[~mask] = 2.0 * np.pi * Rc * Rc / self.volume

        else:
            raise ValueError(f"Unknown exxdiv_treatment: {self.exxdiv_treatment}")

        self._kernel_cache[key] = kern
        return kern

    def _apply_fock_full(self, psi_in):
        if self._occ_states is None or len(self._occ_states) == 0:
            out = psi_in.copy()
            out.data[:] = 0.0
            return out

        psi_in_r = psi_in.to_r()
        psi_in_r *= self._scale

        vx_psi_r = psi_in_r.copy()
        vx_psi_r.data[:] = 0.0

        for st in self._occ_states:
            if st.occ_wt == 0.0:
                continue

            q = self.k_cryst - st.k_cryst
            rho_r = self._conj_mul_wr(st.psi_r, psi_in_r)
            rho_G = rho_r.to_g()

            kern_q = self._kernel_for_q(q)
            np.multiply(rho_G.data, kern_q, out=rho_G.data)

            v_mn_r = rho_G.to_r()
            v_mn_r *= self._scale

            prod_r = self._mul_wr(st.psi_r, v_mn_r)
            vx_psi_r.data[:] += (-0.5 * float(st.occ_wt)) * prod_r.data

        return vx_psi_r.to_g()

    def fock_energy(self, psi, occ):
        # Include the BZ weight for this k-point.
        return self.k_weight_bz * super().fock_energy(psi, occ)

    def energy(self):
        Ex = 0.0

        if self._psi_occ_r is None or len(self._psi_occ_r) == 0:
            return 0.0
        if self._occ_states is None or len(self._occ_states) == 0:
            return 0.0

        for fi, psi_i_r in zip(self._occ, self._psi_occ_r):
            if fi == 0.0:
                continue

            fi_wt = self.k_weight_bz * float(fi)
            for st in self._occ_states:
                if st.occ_wt == 0.0:
                    continue

                q = self.k_cryst - st.k_cryst
                rho_r = self._conj_mul_wr(st.psi_r, psi_i_r)
                rho_G = rho_r.to_g()
                Y2 = np.abs(rho_G.data) ** 2

                kern_q = self._kernel_for_q(q)
                Ex += fi_wt * float(st.occ_wt) * np.sum(kern_q * Y2)

        Ex *= -0.25
        return float(np.real(Ex))
