# Installation Instructions for QuantumMASALA

These installation instructions require `conda`. I you don't have a functioning `conda` installation, we recommend installing it by following these quick-install instructions:
https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install


## Installing on Mac

```
conda create -n qtm python=3.11
conda activate qtm
conda install pyfftw pylibxc mpi4py cython pybind11
```
For optimal performance, we recommend the setting the relevant `..._NUM_THREADS` environment variables to 1:
```
conda env config vars set OMP_NUM_THREADS=1
conda env config vars set VECLIB_MAXIMUM_THREADS=1
conda env config vars set OPENBLAS_NUM_THREADS=1
```
Reactivate the environment for the changes to take place.
```
conda deactivate
conda activate qtm
```
Inside the `QuantumMASALA` root directory, execute the following to complete the installation. 
> [!NOTE]
> Please verify that that the `python` command points to the environment's python installation by running `which python`. If it does not point to the right python installation, we recommend specifying the full `python` path (i.e. `$CONDAPREFIX/bin/python`) for the following commands.
```
python -m pip install -e .
```

Test the installation by running the following example: (Replace `10` with the number of cores)
```
cd examples/dft-fe
mpirun -np 10 python fe_scf.py
```

## Installing on Linux

```
conda create -n qtm python=3.11
conda activate qtm
conda install -c conda-forge pyfftw pylibxc mpi4py
```
For optimal performance, we recommend the setting the `OMP_NUM_THREADS` environment variable to 1:
```
conda env config vars set OMP_NUM_THREADS=1
```
Reactivate the environment for the changes to take place.
```
conda deactivate
conda activate qtm
```
If you are working on an Intel system, we recommend using `mkl_fft` for optimal performance.
```
conda install -c conda-forge mkl_fft
```
Inside the `QuantumMASALA` root directory, execute the following to complete the installation. 
> [!NOTE]
> Please verify that that the `python` command points to the environment's python installation by running `which python`. If it does not point to the right python installation, we recommend specifying the full `python` path (i.e. `$CONDAPREFIX/bin/python`) for the following commands.
```
python -m pip install -e .
```

Test the installation by running the following example: (Replace `10` with the number of cores)
```
cd examples/dft-fe
mpirun -np 10 python fe_scf.py
```
