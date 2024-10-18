from dataclasses import dataclass
from pprint import pformat
from typing import List

from qtm.interfaces.bgw.inp import read_epsilon_inp


@dataclass
class Epsinp:
    """
    Stores ``epsilon.inp`` data.
    Contains two kinds of keywords:
    - Parameters:   Have value and type
    - Options:      Have no value, if set in epsilon.inp, their value is True, by default they are initialized to False

    The doctrings code was generated from html to md conversion of [http://manual.berkeleygw.org/3.0/epsilon-keywords/]
    """

    # Parameters:

    qpts: List[float] = None
    """Expecting shape (numqpt,1).
    qx qy qz.
    """
    scale_factor: List[float] = None
    """
    scale\_factor is for specifying values such as 1/3 .
    """

    is_q0: List[bool] = None
    """is\_q0 = 0 for regular, non-zero q-vectors (read val WFNs from `WFN`) is\_q0 = 1 for a small q-vector in semiconductors (read val WFNs from `WFNq`) is\_q0 = 2 for a small q-vector in metals (read val WFNs from `WFN`) if present the small q-vector should be first in the list You can generate this list with `kgrid.x`: just set the shifts to zero and use same grid numbers as for `WFN`. Then replace the zero vector with q0.
    """

    frequency_dependence_method: int = None
    """Full frequency dependence method for the polarizability, if [`frequency_dependence`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_dependence)\==2:
    *   0: Real-axis formalism, Adler-Wiser formula.
    *   1: Real-axis formalism, spectral method (PRB 74, 035101, (2006))
    *   2: Contour-deformation formalism, Adler-Wiser formula.
    """
    broadening: float = None
    """Broadening parameter for each val->cond transition. It should correspond to the energy resolution due to k-point sampling, or a number as small as possible if you have a molecule. The default value is 0.1 for method 0 and 1, and 0.25 for method 2.
    """
    init_frequency: float = None
    """Lower bound for the linear frequency grid.
    """
    frequency_low_cutoff: float = None
    """Upper bound for the linear frequency grid. For methods 0 and 1, it is also the lower bound for the non-uniform frequency grid. Should be larger than the maximum transition, i.e., the energy difference between the highest conduction band and the lowest valence band. The default is 200 eV for method 0 and 1, and 10 eV for method 2. For method 2, you can increase frequency\_low\_cutoff if you wish to use the Sigma code and look into QP states deep in occupied manifold or high in the unoccupied manifold.
    """
    delta_frequency: float = None
    """Frequency step for full-frequency integration for the linear grid Should be converged (the smaller, the better). For molecules, delta\_frequency should be the same as broadening. Defaults to the value of broadening.
    """
    number_imaginary_freqs: int = None
    """Number of frequencies on the imaginary axis for method 2.
    """
    frequency_high_cutoff: float = None
    """Upper limit of the non-uniform frequency grid. Defaults to 4\*[`frequency_low_cutoff`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_low_cutoff)
    """
    delta_frequency_step: float = None
    """Increase in the frequency step for the non-uniform frequency grid.
    """
    delta_sfrequency: float = None
    """Frequency step for the linear grid for the spectral function method. Defaults to [`delta_frequency`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#delta_frequency).
    """
    delta_sfrequency_step: float = None
    """Increase in frequency step for the non-uniform grid for the spectral function method. Defaults to [`delta_frequency_step`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#delta_frequency_step)
    """
    sfrequency_low_cutoff: float = None
    """Upper bound for the linear grid for the spectral function method and lower bound for the non-uniform frequency grid. Defaults to [`frequency_low_cutoff`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_low_cutoff)
    """
    sfrequency_high_cutoff: float = None
    """Upper limit of the non-uniform frequency grid for the spectral function method. Defaults to [`frequency_low_cutoff`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_low_cutoff)
    """
    """### Static subspace approximation
    Input parameters controlling the full-frequency static subspace approximation method. The method speeds up full-frequency calculations by expanding the frequency-dependent part of the polarizability in the subspace basis formed by the lowest eigenvectors of the static polarizability (see [Phys. Rev. B 99, 125128, 2019](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.125128)). The method is implemented for the case of full-frequency contour-deformation formalism, i.e., [`frequency_dependence`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_dependence)\==2 and [`frequency_dependence_method`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_dependence_method)\==2.
    """
    chi_eigenvalue_cutoff: float = None
    """Activate the static subspace approximation and define the screening threshold for the eigenvalues of the static polarizability.
    """
    nbasis_subspace: int = None
    """Define a maximum fixed number of eigenvectors to be used (recommended). Set for example to 25% of the number of G-vectors employed for the expansion of chi0.
    """
    evs: float = None
    """"""
    ev0: float = None
    """"""
    evdel: float = None
    """"""
    ecs: float = None
    """"""
    ec0: float = None
    """"""
    ecdel: float = None
    """"""
    cvfit: List[int] = None
    """### Truncation schemes for the Coulomb potential
    Since BerkerleyGW is a plane-wave-based code, one must truncate the Coulomb potential to avoid spurious interactions between repeated supercells when dealing with systems with reduced dimensionality. Make sure you understand how to setup your mean-field calculation so that the supercell is large enough to perform a truncation of the Coulomb potential.
    """
    coulomb_truncation_radius: float = None
    """This specifies the radius of for spherical truncation, in Bohr, so that the Coulomb potential v(r) is zero for r larger than these values. This flag is to be used together with [`spherical_truncation`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#spherical_truncation).
    """

    """### Misc. parameters
    """
    epsilon_cutoff: float = None
    """Energy cutoff for the dielectric matrix, in Ry. The dielectric matrix 𝜺_GG' will contain all G-vectors with kinetic energy |q+G|^2 up to this cutoff.
    """
    number_bands: int = None
    """Total number of bands (valence+conduction) to sum over. Defaults to the number of bands in the WFN file minus 1.
    """
    frequency_dependence: int = None
    """This flags specifies the frequency dependence of the inverse dielectric matrix:
    *   Set to 0 to compute the static inverse dielectric matrix (default).
    *   Set to 2 to compute the full frequency dependent inverse dielectric matrix.
    *   Set to 3 to compute the two frequencies needed for Godby-Needs GPP model.
    """
    plasma_freq: float = None
    """Plasma frequency (eV) needed for the contour-deformation method (i.e., [`frequency_dependence`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_dependence)\==2). The exact value is unimportant, especially if you have enough imaginary frequency points. We recommend you keep this value fixed at 2 Ry.
    """
    imaginary_frequency: float = None
    """For [`frequency_dependence`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#frequency_dependence)\==3, the value of the purely imaginary frequency, in eV:
    """
    full_chi_conv_log: int = None
    """Logging convergence of the head & tail of polarizability matrix with respect to conduction bands:
    *   Set to -1 for no convergence test
    *   Set to 0 for the 5 column format including the extrapolated values (default).
    *   Set to 1 for the 2 column format, real part only.
    *   Set to 2 for the 2 column format, real and imaginary parts.
    """
    fermi_level: float = None
    """Specify the Fermi level (in eV), if you want implicit doping Note that value refers to energies _after_ scissor shift or eqp corrections. See also [`fermi_level_absolute`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#fermi_level_absolute) and [`fermi_level_relative`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#fermi_level_relative) to control the meaning of the Fermi level.
    The Fermi level in keyword [`fermi_level`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#fermi_level) can be treated as an absolute value or relative to that found from the mean field (default)
    """
    verbosity: int = None
    """Verbosity level, options are:
    *   1: default
    *   2: medium - info about k-points, symmetries, and eqp corrections.
    *   3: high - full dump of the reduced and unfolded k-points.
    *   4: log - log of various function calls. Use to debug code.
    *   5: debug - extra debug statements. Use to debug code.
    *   6: max - only use if instructed to, severe performance downgrade. Note that verbosity levels are cumulative. Most users will want to stick with level 1 and, at most, level 3. Only use level 4+ if debugging the code.
    """
    number_valence_pools: int = None
    """Number of pools for distribution of valence bands The default is chosen to minimize memory in calculation
    By default, the code computes the polarizability matrix, constructs the dielectric matrix, inverts it and writes the result to file epsmat. Use keyword skip\_epsilon to compute the polarizability matrix and write it to file chimat. Use keyword skip\_chi to read the polarizability matrix from file chimat, construct the dielectric matrix, invert it and write the result to file epsmat.
    """
    nfreq_group: int = None
    """(Full Frequency only) Calculates several frequencies in parallel. No "new" processors are used here, the chi summation is simply done in another order way to decrease communication. This also allows the inversion of multiple dielectric matrices simultaneously via ScaLAPACK, circumventing ScaLAPACK's scaling problems. Can be very efficient for system with lots of G-vectors and when you have many frequencies. In general gives speedup. However, in order to calculate N frequencies in parallel, the memory to store `pol%gme` is currently multiplied by N as well.
    'unfolded BZ' is from the kpoints in the WFN file 'full BZ' is generated from the kgrid parameters in the WFN file See comments in Common/checkbz.f90 for more details
    """
    qgrid: List[int] = None
    """Q-grid for the epsmat file. Defaults to the WFN k-grid.
    """

    # Options:

    write_subspace_epsinv: bool = False
    """Write frequency-dependent epsilon matrices in `eps0mat[.h5]` and `epsmat[.h5]` files using the subspace basis instead of the full G-vector basis (recommended). This flag need to be specified to use the full-frequency static subspace approximation in sigma.
    """
    subspace_dont_keep_full_eps_omega0: bool = False
    """Discharge the output of the full static epsilon matrix in the `eps0mat[.h5]` and `epsmat[.h5]` files.
    """
    subspace_use_elpa: bool = False
    """If ELPA library is linked, specify to use ELPA for the diagonalization of the static polarizability.
    """
    dont_keep_fix_buffers: bool = False
    """Using [`comm_nonblocking_cyclic`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#comm_nonblocking_cyclic) force the algorithm to use buffers for communication with variable size, mainly used for debugging.
    """
    sub_collective_eigen_redistr: bool = False
    """Replicate eigenvectors using collective operation in the basis transformation step, performs in general worse than point to point, mainly used for debugging.
    """
    """### Scissors operator
    Scissors operator (linear fit of the quasiparticle energy corrections) for the bands in `WFN` and `WFNq`. For valence-band energies:
    *   `ev_cor = ev_in + evs + evdel * (ev_in - ev0)`
    For conduction-band energies:
    *   `ec_cor = ec_in + ecs + ecdel * (ec_in - ec0)`
    Defaults is zero for all entries, i.e., no scissors corrections. `evs`, `ev0`, `ecs`, `ec0` are in eV. If you have `eqp.dat` and `eqp_q.dat` files this information is ignored in favor of the eigenvalues in eqp.dat and eqp\_q.dat. One can specify all parameters for scissors operator in a single line with `cvfit evs ev0 evdel ecs ec0 ecdel`
    """
    cell_box_truncation: bool = False
    """Truncate the Coulomb potential based on the Wigner-Seitz cell. This is the recommended truncation for 0D systems.
    """
    cell_wire_truncation: bool = False
    """Truncation scheme for 1D systems, such as carbon nanotubes. The z direction is assumed to be periodic, and x and y confined.
    """
    cell_slab_truncation: bool = False
    """Truncation scheme for 2D systems, such as graphene or monolayer MoS2. The z direction is assumed to be confined, and x and y periodic.
    """
    spherical_truncation: bool = False
    """Truncate the Coulomb potential based on an analytical scheme. This is ok for quasi-spherical systems, such as CH4 molecule or C60, but the [`cell_box_truncation`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#cell_box_truncation) is the most general and recommended scheme. When using spherical truncation, you must also specify the radius for the truncation in [`spherical_truncation`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#spherical_truncation).
    """
    use_wfn_hdf5: bool = False
    """Read input wavefunction files in HDF5 format: `WFN.h5` instead of `WFN` and, if required, `WFNq.h5` instead of `WFNq`.
    """
    comm_nonblocking_cyclic: bool = False
    """Within [`gcomm_matrix`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#gcomm_matrix), employs a non-blocking cyclic communication scheme overlapping computation and communication in the evaluation of the polarizability (drastically reduce the time spent in communication for large runs, but require more memory).
    """
    fermi_level_absolute: bool = False
    """"""
    fermi_level_relative: bool = False
    """"""
    dont_use_hdf5: bool = False
    """Read from traditional simple binary format for epsmat/eps0mat instead of HDF5 file format. Relevant only if code is compiled with HDF5 support.
    """
    eqp_corrections: bool = False
    """Set this to use eigenvalues in eqp.dat and eqp\_q.dat If not set, these files will be ignored.
    """
    write_vcoul: bool = False
    """Write the bare Coulomb potential v(q+G) to file
    """
    gcomm_matrix: bool = False
    """Matrix Element Communication Method (Chi Sum Comm). Default is [`gcomm_matrix`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#gcomm_matrix) which is good if nk,nc,nv\>nmtx,nfreq. If nk,nc,nv<nfreq,nmtx (nk,nv<nfreq since ncâˆ¼nmtx), use gcomm\_elements. Only [`gcomm_elements`](http://manual.berkeleygw.org/3.0/epsilon-keywords/#gcomm_elements) is supported with the spectral method.
    """
    gcomm_elements: bool = False
    """"""
    skip_epsilon: bool = False
    """"""
    skip_chi: bool = False
    """"""
    fullbz_replace: bool = False
    """Replace unfolded BZ with full BZ
    """
    fullbz_write: bool = False
    """Write unfolded BZ and full BZ to files
    """
    degeneracy_check_override: bool = False
    """The requested number of bands cannot break degenerate subspace Use the following keyword to suppress this check Note that you must still provide one more band in wavefunction file in order to assess degeneracy
    """
    no_min_fftgrid: bool = False
    """Instead of using the RHO FFT box to perform convolutions, we automatically determine (and use) the smallest box that is compatible with your epsilon cutoff. This also reduces the amount of memory needed for the FFTs. Although this optimization is safe, you can disable it by uncommenting the following line:
    """
    restart: bool = False
    """Use this flag if you would like to restart your Epsilon calculation instead of starting it from scratch. Note that we can only reuse q-points that were fully calculated. This flag is ignored unless you are running the code with HDF5.
    """
    subsample: bool = False
    """Use this option to generate an eps0mat file suitable for subsampled Sigma calculations. The only thing this flag does is to allow a calculation with more than one qâ†’0 points without complaining.
    """

    @classmethod
    def from_epsilon_inp(cls, filename: str):
        """Read the data from ``epsilon.inp`` file.

        Parameters
        ----------
        filename : str
            Path to the ``epsilon.inp`` file to be read.
        """
        epsinp = Epsinp()
        eps_nt = read_epsilon_inp(filename)

        # Load Parameters
        for field in eps_nt._fields:
            if field in epsinp.__dir__():
                epsinp.__setattr__(field, eps_nt.__getattribute__(field))

        # Load Options
        for option in eps_nt.options:
            if option in eps_nt.__dir__():
                epsinp.__setattr__(option, True)

        return epsinp

    def __repr__(self):
        res = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_") and value is not None and value is not False:
                res += attr + ":\t"
                res += pformat(value, compact=True) + "\n"
        return res


if __name__ == "__main__":
    epsinp = Epsinp()
    print(epsinp.verbosity)

    epsinp = Epsinp.from_epsilon_inp(
        "../scripts/results/si_4_10_ryd_printing/cohsex/epsilon.inp"
    )
