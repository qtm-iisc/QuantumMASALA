from typing import List
from pprint import pformat
from dataclasses import dataclass


from quantum_masala.gw.io_bgw.inp import read_sigma_inp

@dataclass
class Sigmainp():
    """
    Stores ``sigma.inp`` data
    Two kinds of attributes:
    - Parameters: have value and type
    - Options: have no value. If the option has been set in ``sigma.inp``, their value is True. By default they are initialized to False
    
    The docstrings were generated from html to md conversion of [http://manual.berkeleygw.org/3.0/sigma-keywords/]
    """

    band_index_min : int = None
    """Minimum band index for diagonal matrix elements (n=mn=m) of \\Sigma\\Sigma to be computed.
    """
    band_index_max : int = None
    """Maximum band index for diagonal matrix elements (n=mn=m) of \\Sigma\\Sigma to be computed.
    """
    kpts : List[float] = None
    """Kpoints: kx ky kz. 
    Expecting shape (nkpts, 2).
    """
    scale_factor : List[float] = None 
    """
    scale\_factor is for specifying values such as 1/3 .
    """

    sigma_matrix : List[int] = None
    """Alternatively, select a specific value of l and let n and m vary in the range from band\_index\_min to band\_index\_max Set l to 0 to skip the off-diagonal calculation (default) If l = -1 then l\_i is set to n\_i (i = 1 ... noffdiag) i.e. each row is computed at different eigenvalue If l = -2 then l\_i is set to m\_i (i = 1 ... noffdiag) i.e. each column is computed at different eigenvalue For l > 0, all elements are computed at eigenvalue of band l. Set t to 0 for the full matrix (default) or to -1/+1 for the lower/upper triangle
    """

    """### Frequency grid used to evaluate the self-energy
    In order to evaluate \\Sigma(\\omega)\\Sigma(\\omega), you have to specify a frequency grid for the frequencies \\omega\\omega. You can control where to center/shift the grid.
    """
    frequency_grid_shift : int = None
    """Use this flag to control where the center of the frequency grid is. The options are:
    *   0 if you don't want any shift, i.e., \\omega\\omega is an absolute energy
    *   1 if you want to shift the first frequency \\omega=0\\omega=0 to the Fermi energy (this was the default behavior in `BerkeleyGW-1.0`)
    *   2 if you want to center the frequency grid around each mean-field QP energy (default)
    """
    delta_frequency_eval : float = None
    """"""
    max_frequency_eval : float = None
    """"""
    init_frequency_eval : float = None
    """If [`frequency_grid_shift`](#frequency_grid_shift) is 0 or 1, specify the initial frequency [`init_frequency_eval`](#init_frequency_eval) (before the shift), in eV, the frequency spacing [`delta_frequency_eval`](#delta_frequency_eval), in eV, and the number of frequency points [`number_frequency_eval`](#number_frequency_eval).
    """
    number_frequency_eval : int = None
    """### Static subspace approximation
    Within the contour deformation formalism ([`frequency_dependence`](#frequency_dependence)\==2 and [`frequency_dependence_method`](#frequency_dependence_method)\==2), this parameters activate the full-frequency static subspace approximation method in sigma. The full-frequency inverse dielectric matrix calculated by epsilon need to be computed using the static subspace method (set `chi_eigenvalue_cutoff` in [`epsilon.inp`](../epsilon-keywords/)). For the method to be effective the epsilon matrices have to be written using the subspace basis (use `write_subspace_epsinv` in [`epsilon.inp`](../epsilon-keywords/)). The implementation is different than the standard CD, making use of zgemm/dgemm calls.
    """
    finite_difference_form : int = None
    """This flag controls the finite difference form for numerical derivative of \\Sigma(\\omega)\\Sigma(\\omega):
    *   grid = -3 : Calculate Sigma(w) on a grid, using the same frequency grid as in full-frequency calculations.
    *   none = -2 : dSigma/dE = 0 \[skip the expansion\].
    *   backward = -1 : dSigma/dE = (Sigma(Ecor) - Sigma(Ecor-dE)) / dE
    *   central = 0 : dSigma/dE = (Sigma(Ecor+dE) - Sigma(Ecor-dE)) / (2dE)
    *   forward = 1 : dSigma/dE = (Sigma(Ecor+dE) - Sigma(Ecor)) / dE
    *   default = 2 : forward for diagonal and none for off-diagonal
    """
    finite_difference_spacing : float = None
    """Finite difference spacing given in eV (defaults to 1.0)
    """
    """### Options for Hartree-Fock & hybrid calculations
    For Hartree-Fock/hybrid functional, no `epsmat`/`eps0mat` files are needed. Instead provide a list of q-points and the grid size. The list of q-points should not be reduced with time reversal symmetry - because BerkeleyGW never uses time reversal symmetry to unfold the q/k-points. Instead, inversion symmetry does the job in the real version of the code. You can generate this list with kgrid.x: just set the shifts to zero and use same grid numbers as for WFN\_inner.
    """
    qgrid : List[int] = None
    """The regular grid of q-points corresponding to the list.
    """
    """### Scissors operator
    Scissors operator (linear fit of the quasiparticle energy corrections) for the bands in `WFN` and `WFNq`. For valence-band energies:
    *   `ev_cor = ev_in + evs + evdel * (ev_in - ev0)`
    For conduction-band energies:
    *   `ec_cor = ec_in + ecs + ecdel * (ec_in - ec0)`
    Defaults is zero for all entries, i.e., no scissors corrections. `evs`, `ev0`, `ecs`, `ec0` are in eV. If you have `eqp.dat` and `eqp_q.dat` files this information is ignored in favor of the eigenvalues in eqp.dat and eqp\_q.dat. One can specify all parameters for scissors operator in a single line with `cvfit evs ev0 evdel ecs ec0 ecdel`
    """
    evs : float = None
    """"""
    ev0 : float = None
    """"""
    evdel : float = None
    """"""
    ecs : float = None
    """"""
    ec0 : float = None
    """"""
    ecdel : float = None
    """"""
    cvfit : List[int] = None
    """### Screening type
    How does the screening of the system behaves? (default=[`screening_semiconductor`](#screening_semiconductor)) BerkeleyGW uses this information to apply a different numerical procedure to computing the diverging q\\rightarrow 0q\\rightarrow 0 contribution to the screened Coulomb potential W\_{GG'}(q)W\_{GG'}(q). These models are not used in Hartree-Fock calculations.
    """
    coulomb_truncation_radius : float = None
    """This specifies the radius of for spherical truncation, in Bohr, so that the Coulomb potential v(r)v(r) is zero for rr larger than these values. This flag is to be used together with [`spherical_truncation`](#spherical_truncation).
    """
    
    """### Misc. parameters
    """
    screened_coulomb_cutoff : float = None
    """Energy cutoff for the screened Coulomb interaction, in Ry. The screened Coulomb interaction W\_{GG'}(q)W\_{GG'}(q) will contain all G-vectors with kinetic energy |q+G|^2|q+G|^2 up to this cutoff. Default is the epsilon\_cutoff used in the epsilon code. This value cannot be larger than the epsilon\_cutoff or the bare\_coulomb\_cutoff.
    """
    bare_coulomb_cutoff : float = None
    """Energy cutoff for the bare Coulomb interaction, in Ry. The bare Coulomb interaction v(G+q)v(G+q) will contain all G-vectors with kinetic energy |q+G|^2|q+G|^2 up to this cutoff. Default is the WFN cutoff.
    """
    number_bands : int = None
    """Total number of bands (valence+conduction) to sum over. Defaults to the number of bands in the WFN file minus 1.
    The range of spin indices for which Sigma is calculated The default is the first spin component
    """
    spin_index_min : int = None
    """"""
    spin_index_max : int = None
    """"""
    cell_average_cutoff : float = None
    """Cutoff energy (in Ry) for averaging the Coulomb Interaction in the mini-Brillouin Zones around the Gamma-point without Truncation or for Cell Wire or Cell Slab Truncation.
    """
    frequency_dependence : int = None
    """Frequency dependence of the inverse dielectric matrix.
    *   Set to -1 for the Hartree-Fock and hybrid functional approximation.
    *   Set to 0 for the static COHSEX approximation.
    *   Set to 1 for the Generalized Plasmon Pole model (default).
    *   Set to 2 for the full frequency dependence.
    *   Set to 3 for the Generalized Plasmon Pole model in the Godby-Needs flavor. Note: does not work with parallelization in frequencies.
    """
    frequency_dependence_method : int = None
    """Full frequency dependence method for the polarizability. set to 0 for the real-axis integration method. set to 2 for the contour-deformation method (default).
    """
    cd_integration_method : int = None
    """For contour deformation calculations, specify the integration method on the imaginary axis (default is 0). Options are:
    *   0 for piecewise constant matrix elements centered at each imaginary frequency
    *   2 for piecewise quadratic matrix elements over each integration segment
    *   3 for cubic Hermite matrix elements over each integration segment
    """
    invalid_gpp_mode : int = None
    """What should we do when we perform a HL-GPP or a GN-GPP calculations and we find an invalid mode frequency with \\omega\_{GG'}^2 < 0\\omega\_{GG'}^2 < 0 Options are: -1: Default, same as 3. 0: Skip invalid mode and ignore its contribution to the self energy. 1: "Find" a purely complex mode frequency. This was the default behavior in BGW-1.x. 2: Set the mode frequency to a fixed value of 2 Ry. 3: Treat that mode within the static COHSEX approximation (move frequency to infinity).
    """
    full_ch_conv_log : int = None
    """Logging the convergence of the CH term with respect to the number of bands in the output file "ch\_converge.dat". Options are:
    *   0 to log only the real part of VBM, CBM and the gap (default).
    *   1 to log the real and imag. parts of all bands for which we compute Sigma.
    """
    bare_exchange_fraction : float = None
    """Fraction of bare exchange. Set to 1.0 if you use the exchange-correlation matrix elements read from file vxc.dat. Set to 1.0 for local density functional, 0.0 for HF, 0.75 for PBE0, 0.80 for B3LYP if you use the local part of the exchange-correlation potential read from file VXC. For functionals such as HSE whose nonlocal part is not some fraction of bare exchange, use vxc.dat and not this option. This is set to 1.0 by default.
    """
    gpp_broadening : float = None
    """Broadening for the energy denominator in CH and SX within GPP. If it is less than this value, the sum is better conditioned than either CH or SX directly, and will be assigned to SX while CH = 0. This is given in eV, the default value is 0.5
    """
    gpp_sexcutoff : float = None
    """Cutoff for the poles in SX within GPP. Divergent contributions that are supposed to sum to zero are removed. This is dimensionless, the default value is 4.0
    """
    fermi_level : float = None
    """Specify the Fermi level (in eV), if you want implicit doping Note that value refers to energies _after_ scissor shift or eqp corrections. See also [`fermi_level_absolute`](#fermi_level_absolute) and [`fermi_level_relative`](#fermi_level_relative) to control the meaning of the Fermi level.
    The Fermi level in keyword [`fermi_level`](#fermi_level) can be treated as an absolute value or relative to that found from the mean field (default)
    """
    verbosity : int = None
    """Verbosity level, options are:
    *   1: default
    *   2: medium - info about k-points, symmetries, and eqp corrections.
    *   3: high - full dump of the reduced and unfolded k-points.
    *   4: log - log of various function calls. Use to debug code.
    *   5: debug - extra debug statements. Use to debug code.
    *   6: max - only use if instructed to, severe performance downgrade. Note that verbosity levels are cumulative. Most users will want to stick with level 1 and, at most, level 3. Only use level 4+ if debugging the code.
    """
    evs_outer : float = None
    """"""
    ev0_outer : float = None
    """"""
    evdel_outer : float = None
    """"""
    ecs_outer : float = None
    """"""
    ec0_outer : float = None
    """"""
    ecdel_outer : float = None
    """"""
    cvfit_outer : List[int] = None
    """One can specify these parameters in a single line as (evs\_outer ev0\_outer evdel\_outer ecs\_outer ec0\_outer ecdel\_outer)
    """
    avgpot : float = None
    """The average potential on the faces of the unit cell in the non-periodic directions for the bands in WFN\_inner This is used to correct for the vacuum level The default is zero, avgpot is in eV
    """
    avgpot_outer : float = None
    """The average potential on the faces of the unit cell in the non-periodic directions for the bands in WFN\_outer This is used to correct for the vacuum level. Has no effect if WFN\_outer is not supplied. The default is zero, avgpot\_outer is in eV
    """
    number_sigma_pools : int = None
    """Number of pools for parallel sigma calculations The default is chosen to minimize memory in calculation
    """
    exact_static_ch : int = None
    """Dealing with the convergence of the CH term. Set to 0 to compute a partial sum over empty bands. Set to 1 to compute the exact static CH. In case of exact\_static\_ch = 1 and frequency\_dependence = 1 (GPP) or 2 (FF), the partial sum over empty bands is corrected with the static remainder which is equal to 1/2 (exact static CH - partial sum static CH), additional columns in sigma\_hp.log labeled ch', sig', eqp0', eqp1' are computed with the partial sum without the static remainder, and ch\_converge.dat contains the static limit of the partial sum. In case of exact\_static\_ch = 0 and frequency\_dependence = 0 (COHSEX), columns ch, sig, eqp0, eqp1 contain the exact static CH, columns ch', sig', eqp0', eqp1' contain the partial sum static CH, and ch\_converge.dat contains the static limit of the partial sum. For exact\_static\_ch = 1 and frequency\_dependence = 0 (COHSEX), columns ch', sig', eqp0', eqp1' are not printed and file ch\_converge.dat is not written. Default is 0 for frequency\_dependence = 1 and 2; 1 for frequency\_dependence = 0; has no effect for frequency\_dependence = -1. It is important to note that the exact static CH answer depends not only on the screened Coulomb cutoff but also on the bare Coulomb cutoff because G-G' for G's within the screened Coulomb cutoff can be outside the screened Coulomb cutoff sphere. And, therefore, the bare Coulomb cutoff sphere is used.
    """
    # Options:

    number_diag : bool = False
    """Number or diagonal matrix elements, i.e., for n=mn=m.
    """
    number_offdiag : bool = False
    """Number of off-diagonal matrix elements
    """
    do_sigma_subspace : bool = False
    """Activate the full-frequency static subspace approximation method in sigma.
    """

    """### Options for the generalized plasmon-pole (GPP) calculations
    The matrix element of the self-energy operator is expanded to first order in the energy around Ecor.
    """
    screening_semiconductor : bool = False
    """Use this flag on gapped system (**default**).
    """
    screening_metal : bool = False
    """Use this flag on metallic system, i.e., constant DOS at the Fermi energy.
    """
    screening_graphene : bool = False
    """Use this flag on systems with vanishing linear DOS at the Fermi level, such as graphene.
    """
    
    """### Truncation schemes for the Coulomb potential
    Since BerkerleyGW is a plane-wave-based code, one must truncate the Coulomb potential to avoid spurious interactions between repeated supercells when dealing with systems with reduced dimensionality. Make sure you understand how to setup your mean-field calculation so that the supercell is large enough to perform a truncation of the Coulomb potential.
    """
    cell_box_truncation : bool = False
    """Truncate the Coulomb potential based on the Wigner-Seitz cell. This is the recommended truncation for 0D systems.
    """
    cell_wire_truncation : bool = False
    """Truncation scheme for 1D systems, such as carbon nanotubes. The zz direction is assumed to be periodic, and xx and yy confined.
    """
    cell_slab_truncation : bool = False
    """Truncation scheme for 2D systems, such as graphene or monolayer MoS2. The zz direction is assumed to be confined, and xx and yy periodic.
    """
    spherical_truncation : bool = False
    """Truncate the Coulomb potential based on an analytical scheme. This is ok for quasi-spherical systems, such as CH4 molecule or C60, but the [`cell_box_truncation`](#cell_box_truncation) is the most general and recommended scheme. When using spherical truncation, you must also specify the radius for the truncation in [`spherical_truncation`](#spherical_truncation).
    """
    use_epsilon_remainder : bool = False
    """Add remainder from tail of epsilon for full frequency.
    """
    use_xdat : bool = False
    """Use precalculated matrix elements of bare exchange from x.dat. The default is not to use them.
    """
    dont_use_vxcdat : bool = False
    """The default behavior is to load the precalculated exchange-correlation matrix elements \\langle n | \\hat{V}\_{xc} | m \\rangle\\langle n | \\hat{V}\_{xc} | m \\rangle from file `vxc.dat`. Use this flag to load the whole exchange-correlation matrix in reciprocal space, V\_{xc}(G)V\_{xc}(G), which should be provided in the file `VXC`.
    """
    use_kihdat : bool = False
    """This flag controls a different way to construct quasiparticle energies It needs kih.dat file generated from pw2bgw.x KIH = Kinetic + Ion + Hartree In this way, we avoid the use of VXC or vxc.dat and it enables BerkeleyGW to interface with many other functionals such as hybrid, metaGGA (including SCAN), etc.
    """
    fermi_level_absolute : bool = False
    """"""
    fermi_level_relative : bool = False
    """"""
    dont_use_hdf5 : bool = False
    """Read from traditional simple binary format for epsmat/eps0mat instead of HDF5 file format. Relevant only if code is compiled with HDF5 support.
    """
    use_wfn_hdf5 : bool = False
    """Read WFN\_inner in HDF5 format (i.e. read from WFN\_inner.h5).
    Scissors operator (linear fit of the quasiparticle energy corrections) for the bands in WFN\_outer. Has no effect if WFN\_outer is not supplied. For valence-band energies: ev\_cor = ev\_in + evs\_outer + evdel\_outer (ev\_in - ev0\_outer) For conduction-band energies: ec\_cor = ec\_in + ecs\_outer + ecdel\_outer (ec\_in - ec0\_outer) Defaults below. evs\_outer, ev0\_outer, ecs\_outer, ec0\_outer are in eV
    """
    eqp_corrections : bool = False
    """Set this to use eigenvalues in eqp.dat If not set, this file will be ignored.
    """
    eqp_outer_corrections : bool = False
    """Set this to use eigenvalues in eqp\_outer.dat If not set, this file will be ignored. Has no effect if WFN\_outer is not supplied.
    """
    write_vcoul : bool = False
    """Write the bare Coulomb potential v(q+G)v(q+G) to file
    """
    tol_degeneracy : bool = False
    """Threshold for considering bands degenerate, for purpose of making sure all of degenerate subspaces are included, for band-averaging, and for setting offdiagonals to zero by symmetry. (Ry)
    'unfolded BZ' is from the kpoints in the WFN\_inner file 'full BZ' is generated from the kgrid parameters in the WFN\_inner file See comments in Common/checkbz.f90 for more details
    """
    fullbz_replace : bool = False
    """Replace unfolded BZ with full BZ
    """
    fullbz_write : bool = False
    """Write unfolded BZ and full BZ to files
    """
    degeneracy_check_override : bool = False
    """The requested number of bands cannot break degenerate subspace Use the following keyword to suppress this check Note that you must still provide one more band in wavefunction file in order to assess degeneracy
    The sum over q-points runs over the full Brillouin zone. For diagonal matrix elements between non-degenerate bands and for spherically symmetric Coulomb potential (no truncation or spherical truncation), the sum over q-points runs over the irreducible wedge folded with the symmetries of a subgroup of the k-point. The latter is the default. In both cases, WFN\_inner should have the reduced k-points from an unshifted grid, i.e. same as q-points in Epsilon. With no\_symmetries\_q\_grid, any calculation can be done; use\_symmetries\_q\_grid is faster but only diagonal matrix elements of non-degenerate or band-averaged states can be done.
    """
    no_symmetries_q_grid : bool = False
    """"""
    use_symmetries_q_grid : bool = False
    """"""
    dont_symmetrize : bool = False
    """If no\_symmetries\_q\_grid is used, this flag skips the averaging of the degenerate subspace. This might be useful for treating accidental degeneracies.
    """
    no_symmetries_offdiagonals : bool = False
    """Off-diagonal elements are zero if the two states belong to different irreducible representations. As a simple proxy, we use the size of the degenerate subspaces of the two states: if the sizes are different, the irreps are different, and the matrix element is set to zero without calculation. Turn off this behavior for testing by setting flag below. Using WFN\_outer effectively sets no\_symmetries\_offdiagonals.
    Rotation of the k-points may bring G-vectors outside of the sphere. Use the following keywords to specify whether to die if some of the G-vectors fall outside of the sphere. The default is to die. Set to die in case screened\_coulomb\_cutoff = epsilon\_cutoff. Set to ignore in case screened\_coulomb\_cutoff < epsilon\_cutoff.
    """
    die_outside_sphere : bool = False
    """"""
    ignore_outside_sphere : bool = False
    """"""
    skip_averagew : bool = False
    """Do not average W over the minibz, and do not replace the head of eps0mat with averaged value. Only use this option for debugging purposes.
    """
    subsample : bool = False
    """By default, the code reads the dielectric matrix for a single q->0 q-point. The following flag enables the subsampling of Voronoi cell containing Gamma. Your eps0mat file should contain a list of radial q-points (which will not be unfolded by symmetries) instead of a single q->0 point. You should provide a file subweights.dat containing the weights w(q) associated to each subsampled q-point (which will be renormalized so that \\sum w(q)=1). Using this subsampling allows one to accelerate the convergence with respect to the number of q-points, and is especially helpful when dealing with large unit cells, truncated Coulomb potential and birefringent materials.
    """

    def validate(self):
        """Validate the required inputs."""
        assert self.band_index_max != None, "band_index_max is required"
        assert self.band_index_min != None, "band_index_min is required"
        assert len(self.kpts) > 0,          "list of kpoints is required"
            
    
    @classmethod
    def from_sigma_inp(cls, filename:str):
        """Read the data from ``sigma.inp`` file.

        Parameters
        ----------
        filename : str
            Path to the ``sigma.inp`` file to be read.
        """
        sigmainp = Sigmainp()
        sigma_nt = read_sigma_inp(filename)

        # Load Parameters
        for field in sigma_nt._fields:
            if field in sigmainp.__dir__():
                sigmainp.__setattr__(field, sigma_nt.__getattribute__(field))
        
        # Load Options
        for option in sigma_nt.options:
            if option in sigmainp.__dir__():
                sigmainp.__setattr__(option, True)

        sigmainp.validate()

        return sigmainp


    def __repr__(self):
        res=""
        for attr,value in self.__dict__.items():
            if not attr.startswith("_") and value is not None and value is not False:
                res+=attr+":\t"
                # pprint(value,compact=True)
                res+=pformat(value,compact=True)+"\n"
        return res