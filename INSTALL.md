# Installation Instructions for QuantumMASALA

These installation instructions require `conda`. If you don't have a functioning `conda` installation, we recommend installing it by following these quick-install instructions:
https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install


## Installing on Mac

```
conda create -n qtm python=3.11 numpy "libblas=*=*accelerate"
```
> [!NOTE]
> For older versions of python one may need to install numpy separately and the 'conda create -n qtm2 python=3.11 numpy "libblas=*=*accelerate"' may not work. As the numpy that comes with conda is often slow, you can overwrite this numpy with the one that uses Apple's Accelerate framework (Veclib) later (see below). For these older versions, first create the qtm environment.
```
conda create -n qtm python=3.11
```
We then activate the environment and install packages in it.
```
conda activate qtm
conda install -c conda-forge pyfftw pylibxc mpi4py scipy cython pybind11
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
> Please verify that that the `python` command points to the environment's python installation by running `which python`. If it does not point to the right python installation, we recommend specifying the full `python` path (i.e. `$CONDA_PREFIX/bin/python`) for the following commands.
```
python -m pip install -e .
```
> [!NOTE]
> For older versions of python one may need to install numpy separately and the 'conda create -n qtm2 python=3.11 numpy "libblas=*=*accelerate"' may not work. As the numpy that comes with conda is often slow, you can overwrite this numpy with the one that uses Apple's Accelerate framework (Veclib). To install that use the following commands.
```
python -m pip install --no-binary :all: --no-use-pep517 --force numpy
```
To test whether it is using the correct backend libraries, you can check with the following commands.
```
python
import numpy
numpy.show_config()
```

Test the installation by running the following example: (Replace `10` with the number of cores)
```
cd examples/dft-fe
mpirun -np 10 python fe_scf.py
```

## Installing on Linux

### Quick Installation

Inside the `QuantumMASALA` root directory, execute the following:
```conda env create -f env_linux.yml -n qtm```
This will create a new conda environment with the necessary packages installed.
If you are working on an Intel system, we recommend using `mkl_fft` for optimal performance.
```conda install -c conda-forge mkl_fft```

Activate the environment:
```conda activate qtm```

Install QuantumMASALA:
```python -m pip install -e .```


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

Test the installation by running the following example: (Replace `10` with the number of cores)
```
cd examples/dft-fe
mpirun -np 10 python fe_scf.py
```
