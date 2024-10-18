# Installation Instructions for QuantumMASALA

These installation instructions require `conda`. If you don't have a functioning `conda` installation, we recommend installing it by following these quick-install instructions:
https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install


## Installing on Mac

```
conda create -n qtm python=3.12 numpy "libblas=*=*accelerate" pyfftw pylibxc mpi4py scipy cython
```
> [!NOTE]
> For older versions of python one may need to install numpy separately with Apple's accelerate framework. With newer versions of numpy > 2.0 the wheels are automatically built with Apple's Accelerate framework (Veclib).

Activate the environment
```
conda activate qtm
```
If you want to test the numpy and scipy installation (a very good idea), install the following packages
```
conda install -c conda-forge pytest hypothesis meson scipy-tests pooch pyyaml
```
> [!NOTE]
> Please verify that that the `python` command points to the environment's python installation by running `which python`. If it does not point to the right python installation, we recommend specifying the full `python` path (i.e. `$CONDA_PREFIX/bin/python`) for the following commands.

To test numpy and scipy, you can check with the following commands.
```
python
import numpy
numpy.show_config()
numpy.test()
import scipy
scipy.show_config()
scipy.test()
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

```
python -m pip install -e .
```

Test the installation by running the following example: (Replace `2` with the number of cores)
```
cd examples/dft-si
mpirun -np 2 python si_scf.py
```

To perform a complete test, use `pytest` from the main directory which contains the tests folder:
```
python -m pip install pytest
pytest
```


## Installing on Linux

### Quick Installation

Inside the `QuantumMASALA` root directory, execute the following:
```
conda env create -f env_linux.yml -n qtm
```
This will create a new conda environment with the necessary packages installed.
If you are working on an Intel system, we recommend using `mkl_fft` for optimal performance.
```
conda install -c conda-forge mkl_fft
```

Activate the environment:
```
conda activate qtm
```

Install QuantumMASALA:
```
python -m pip install -e .
```


### Manual Installation

```
conda create -n qtm python=3.11
conda activate qtm
conda install -c conda-forge pyfftw pylibxc mpi4py
```
If you are working on an Intel system, we recommend using `mkl_fft` for optimal performance.
```
conda install -c conda-forge mkl_fft
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
Inside the `QuantumMASALA` root directory, execute the following to complete the installation. 
> [!NOTE]
> Please verify that that the `python` command points to the environment's python installation by running `which python`. If it does not point to the right python installation, we recommend specifying the full `python` path (i.e. `$CONDA_PREFIX/bin/python`) for the following commands.
```
python -m pip install -e .
```

Test the installation by running the following example: (Replace `2` with the number of cores)
```
cd examples/dft-si
mpirun -np 2 python si_scf.py
```

To perform a complete test, use `pytest` from the main directory which contains the tests folder:
```
python -m pip install pytest
pytest
```
