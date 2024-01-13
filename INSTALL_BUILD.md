# Instructions for building the prerequisite libraries for QuantumMASALA from scratch

### Conda environment
Create a python environment where we will install our python packages.
```bash
conda create -n qtm python=3.9
conda activate qtm
```

Check the `PYTHONPATH` envronment variable, and modify it if it is not pointing to the current conda environment's python executable.
```bash
conda env config vars set PYTHONPATH=$CONDA_PREFIX/bin/python

# Deactivate and reactivate the conda environment for changes to take effect
conda deactivate
conda activate qtm
```

### Installing NumPy

For version 1.26 onwards, NumPy developers recommend using Meson for building NumPy.
However, the `setup.py` based method of building is still supported.
```bash
python setup.py build
python setup.py install
```
Verify that the build is linked to the intended BLAS and LAPACK backends.
```python
import numpy
numpy.show_config()
```
It is recommended to test the package after installation.
```python
numpy.test('full')
```

### Installing SciPy

version 1.11 onwards, only Meson-based installation is supported.
For now, we have gone to version 1.10.
```bash
python setup.py build
python setup.py install
```
Verify that the build is linked to the intended BLAS and LAPACK backends.
```python
import scipy
scipy.show_config()
```
It is recommended to test the package after installation.
```python
scipy.test('full')
```

> [!WARNING]
> The tests for scipy.sparse.linalg may fail due to known issues with PROPACK and MKL.
> [BUG: Seg. fault in scipy.sparse.linalg tests in PROPACK #15108](https://github.com/scipy/scipy/issues/15108)
> `eigsh` is the only function we use from this module and only in our scipy-based diagonalizer. Therefore, if one finds that the said tests fail, they are suggested to avoid using the said diagonalizer.


### Installing pyFFTW
- Build FFTW as per the instructions given here: https://pyqg.readthedocs.io/en/latest/installation.html#the-hard-way-installing-from-source
- Git clone https://github.com/pyFFTW/pyFFTW.git
> [!IMPORTANT]
> Use Cython version <3, to avoid [known issues](https://github.com/pyFFTW/pyFFTW/issues/362#issue-1868263621).


### Installing pylibxc

- For a system with Intel OneAPI installed:
  ```bash
  cmake -H. -DCMAKE_INSTALL_PREFIX=/home/.../Install -DENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON -DENABLE_FORTRAN=ON -DCMAKE_Fortran_COMPILER=ifort -DCMAKE_C_COMPILER=icx -Bobjdir
  cd objdir && make
  make test
  make install
  ```

- Install pylibxc by calling the following from the libxc root directory:
  ```bash
  python setup.py install
  ```
  Test the installation by runnng the following python code:
  ```python
  # Build functional
  func = pylibxc.LibXCFunctional("gga_c_pbe", "unpolarized")

  # Create input
  inp = {}
  inp["rho"] = np.random.random((3))
  inp["sigma"] = np.random.random((3))

  # Compute
  ret = func.compute(inp)
  for k, v in ret.items():
      print(k, v)
  ```
  The output should look something like the following:
  ```bash
  zk [[-0.06782171 -0.05452743 -0.04663709]]
  vrho [[-0.08349967 -0.0824188  -0.08054892]]
  vsigma [[ 0.00381277  0.00899967  0.01460601]]

  ```