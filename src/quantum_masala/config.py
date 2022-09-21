from typing import TypedDict, Literal, Optional
from importlib.util import find_spec

MPI4PY_INSTALLED = find_spec("mpi4py") is not None
CUPY_INSTALLED = find_spec("cupy") is not None
PRIMME_INSTALLED = find_spec("primme") is not None

if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
    USE_MPI = COMM_WORLD.Get_size() != 1
else:
    USE_MPI = False



class FFTConfig(TypedDict):
    LIB: Literal["NUMPY", "SCIPY", "MKLFFT", "PYFFTW"]
    METHOD: Literal["SLAB", "STICKS"]


class PyFFTWConfig(TypedDict):
    PLANNER_EFFORT: Literal[
        "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"
    ]
    NUM_THREADS: int


class SpFFTConfig(TypedDict):
    NUM_THREADS: int


PYFFTW_CONFIG: PyFFTWConfig = {"PLANNER_EFFORT": "FFTW_PATIENT",
                               "NUM_THREADS": 1
                               }
SPFFT_CONFIG: SpFFTConfig = {"NUM_THREADS": 1}
FFT_CONFIG: FFTConfig = {"LIB": "PYFFTW",
                         "METHOD": "SLAB"
                         }



class SpglibConfig(TypedDict):
    SYMPREC: float
    ANGLE_TOL: float


SPGLIB_CONFIG: SpglibConfig = {"SYMPREC": 1e-5,
                               "ANGLE_TOL": -1.0
                               }



class LibXCConfig(TypedDict):
    APPLY_THR: bool
    LDA_RHO_THR: Optional[float]
    GGA_RHO_THR: Optional[float]
    GGA_SIG_THR: Optional[float]


LIBXC_CONFIG: LibXCConfig = {
    "APPLY_THR": True,
    "LDA_RHO_THR": None,
    "GGA_RHO_THR": None,
    "GGA_SIG_THR": None,
}

LIBXC_MAP: dict[str, str] = {"pbe": "gga_x_pbe gga_c_pbe"}

GPU_MODE = False
