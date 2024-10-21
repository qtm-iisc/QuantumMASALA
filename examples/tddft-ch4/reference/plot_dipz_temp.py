import matplotlib.pyplot as plt
from scipy.constants import eV, c
import numpy as np
from qtm.tddft_gamma.optical import dipole_spectrum
from qtm.constants import ELECTRONVOLT


dip_z = np.load('dipz_temp.npy')

# Alternatively, you can load the dipole moment from the file 'dipz.npy'
# dip_z = np.load('dipz.npy')

dip_z = dip_z[:] - dip_z[0]    # Subtract the initial dipole moment

print("Shape of dipole moment array:", dip_z.shape)
numsteps = dip_z.shape[0]
time_step = 0.1


# Time domain data
au_to_fs = 2.418884326505e-2
time = np.linspace(0, numsteps*time_step, numsteps, endpoint=False) * au_to_fs  # Convert to femtoseconds
print("Time array shape:", time.shape)

# Frequency domain data
dim = 2   # z-component of the dipole moment
en_end = 40 * ELECTRONVOLT

spectrum = dipole_spectrum(dip_z, time_step=time_step, damp_func='gauss', en_end=en_end, en_step=en_end/len(dip_z), damp_fac=5e-3)


# Show the plots together
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(time, dip_z[:,2])
axs[0].set_xlabel("Time (fs)")
axs[0].set_ylabel("Dipole Response (a.u.)")
axs[0].set_title("Dipole Response vs Time")
axs[0].grid()

axs[1].plot(spectrum[0]/ELECTRONVOLT,np.imag(spectrum[1][:,dim]), label='QTM')
axs[1].set_xlabel("Energy (eV)")
axs[1].set_ylabel("Absorption Spectrum (arb. units)")
axs[1].set_title("Absorption Spectrum")
axs[1].legend()

plt.tight_layout()

plt.savefig("dipz.png")
