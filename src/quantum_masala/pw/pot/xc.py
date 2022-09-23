import numpy as np

from pylibxc import LibXCFunctional
import pylibxc.flags as xc_flags

from quantum_masala.core import Rho, DelOperator

from .base import LocalPot

LIBXC_THR_LDA_RHO = 1e-10
LIBXC_THR_GGA_RHO = 1e-6
LIBXC_THR_GGA_SIG = 1e-10


class ExchCorr(LocalPot):
    def __init__(
        self,
        rho: Rho,
        exch_name: str,
        corr_name: str,
    ):
        super().__init__(rho)
        self.deloper = DelOperator(self.grho, self.fft_rho)
        self.numspin = self.rho.numspin

        self.exch_func = LibXCFunctional(
            exch_name, "unpolarized" if self.numspin == 1 else "polarized"
        )
        self.corr_func = LibXCFunctional(
            corr_name, "unpolarized" if self.numspin == 1 else "polarized"
        )

        self.is_grad = sum(
            True
            if xcfunc.get_family()
            in [xc_flags.XC_FAMILY_GGA, xc_flags.XC_FAMILY_HYB_GGA]
            else False
            for xcfunc in [self.exch_func, self.corr_func]
        )

        for xcfunc in [self.exch_func, self.corr_func]:
            if xcfunc.get_family() == xc_flags.XC_FAMILY_LDA:
                xcfunc.set_dens_threshold(LIBXC_THR_LDA_RHO)
            elif xcfunc.get_family() == xc_flags.XC_FAMILY_GGA:
                xcfunc.set_dens_threshold(LIBXC_THR_GGA_RHO)
                xcfunc.set_sigma_threshold(LIBXC_THR_GGA_SIG)

        self.vexch_r = np.empty((self.numspin, *self.grho.grid_shape), dtype="c16")
        self.vcorr_r = np.empty((self.numspin, *self.grho.grid_shape), dtype="c16")
        self.vexch_sum, self.vcorr_sum = None, None
        self.en_exch, self.en_corr = None, None

    @property
    def g(self):
        return self.fft_rho.r2g(self.r)

    @property
    def r(self):
        return self.vexch_r + self.vcorr_r

    @property
    def en(self):
        return self.en_corr + self.en_exch

    @property
    def vxc_sum(self):
        return self.vexch_sum + self.vcorr_sum

    def sync(self):
        self.vexch_r = self.pwcomm.world_comm.Bcast(self.vexch_r)
        self.vexch_sum = self.pwcomm.world_comm.bcast(self.vexch_sum)
        self.en_exch = self.pwcomm.world_comm.bcast(self.en_exch)

        self.vcorr_r = self.pwcomm.world_comm.Bcast(self.vcorr_r)
        self.vcorr_sum = self.pwcomm.world_comm.bcast(self.vcorr_sum)
        self.en_corr = self.pwcomm.world_comm.bcast(self.en_corr)

    def compute(self):
        grid_shape = self.grho.grid_shape
        xc_inp = {
            "rho": np.copy(np.transpose(self.rho.r.real.reshape(self.numspin, -1)), "C")
        }

        sigma_r, grad_rho_r = None, None
        if self.is_grad:
            grad_rho_r = self.deloper.grad(self.rho.g)
            grad_rhoaux_r = self.deloper.grad(self.rho.aux_g)
            sigma_r = [np.sum(grad_rhoaux_r[0] * grad_rhoaux_r[0], axis=0)]
            if self.numspin != 1:
                sigma_r.append(np.sum(grad_rhoaux_r[0] * grad_rhoaux_r[1], axis=0))
                sigma_r.append(np.sum(grad_rhoaux_r[1] * grad_rhoaux_r[1], axis=0))
            sigma_r = np.array(sigma_r)
            xc_inp["sigma"] = np.copy(
                np.transpose(sigma_r.real.reshape(sigma_r.shape[0], -1)), order="C"
            )

        for xcfunc in [self.exch_func, self.corr_func]:
            xcfunc_out = xcfunc.compute(xc_inp)
            zk_r = xcfunc_out["zk"].reshape(grid_shape)
            v_r = (
                np.transpose(xcfunc_out["vrho"])
                .reshape((self.numspin, *grid_shape))
                .astype("c16")
            )
            v_sum = self.rho.integral_rho_f_dv(v_r)
            en = self.rho.integral_rho_f_dv(zk_r)

            if self.is_grad:
                vsig_r = np.transpose(xcfunc_out["vsigma"]).reshape(
                    (sigma_r.shape[0], *grid_shape)
                )
                h_r = np.empty((self.numspin, 3, *grid_shape), dtype="c16")
                if self.numspin == 1:
                    h_r[0] = 2 * vsig_r[0] * grad_rho_r[0]
                else:
                    h_r[0] = 2 * vsig_r[0] * grad_rho_r[0] + vsig_r[1] * grad_rho_r[1]
                    h_r[1] = 2 * vsig_r[2] * grad_rho_r[1] + vsig_r[1] * grad_rho_r[0]
                div_h_r = self.deloper.div(h_r)
                v_r -= div_h_r
                v_sum -= self.rho.integral_rho_f_dv(div_h_r)

            if xcfunc == self.exch_func:
                self.vexch_r[:], self.vexch_sum, self.en_exch = v_r, v_sum, en
            else:
                self.vcorr_r[:], self.vcorr_sum, self.en_corr = v_r, v_sum, en
        self.sync()
