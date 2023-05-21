"""Functions for reading `RHO`, `vxc.dat`, `epsilon.inp`, `sigma.inp`

Core functions:
- read_rho(filename)
- read_vxc(filename)
- read_epsilon_inp(filename)
- read_sigma_inp(filename)
"""

import json
import numpy as np
from typing import Dict, NamedTuple
from quantum_masala.gw.h5_io.h5_to_np_dict import read_input_h5
from collections import namedtuple
from copy import deepcopy

# Common functions ----------------------------------------------------


def dict_to_namedtuple(data, name):
    """Convert dict to namedtuple"""
    temp = deepcopy(data)
    for key in temp:
        if isinstance(temp[key], dict):
            temp[key] = dict_to_namedtuple(temp[key], key)
    nt = namedtuple(name, temp)
    temp_nt = nt(**temp)
    return temp_nt


def read_input_h5_namedtuple(filename, name, **kwargs):
    my_dict = read_input_h5(filename,**kwargs)
    data = dict_to_namedtuple(my_dict, name)
    return data


def read_rho(filename):
    """Read RHO - the fortran binary file containg rho data
    Returns lists of gvecs and list of corresponding rho values (complex for now).
    Comments give the corresponding lines from pw2bgw that wrote out the RHO file.
    We read and ignore irrelevant data.
    """
    from scipy.io import FortranFile
    import numpy as np

    f = FortranFile(filename, "r")

    # 0  WRITE ( unit ) stitle, sdate, stime
    header = "".join(map(chr, f.read_ints("int8")))
    # 1  WRITE ( unit ) nsf, ng_g, ntran, cell_symmetry, nat, ecutrho
    # print(f.read_ints('int32'))
    f.read_record(np.dtype(("<i4", (6))), "f4")
    # 2  WRITE ( unit ) dfftp%nr1, dfftp%nr2, dfftp%nr3
    f.read_ints("int32")
    # 3  WRITE ( unit ) omega, alat, ( ( at ( j, i ), j = 1, nd ), i = 1, nd ), &
    #     ( ( adot ( j, i ), j = 1, nd ), i = 1, nd )
    f.read_reals("double")
    # 4  WRITE ( unit ) recvol, tpiba, ( ( bg ( j, i ), j = 1, nd ), i = 1, nd ), &
    #     ( ( bdot ( j, i ), j = 1, nd ), i = 1, nd )
    f.read_reals("double")
    # 5  WRITE ( unit ) ( ( ( s ( k, j, i ), k = 1, nd ), j = 1, nd ), i = 1, ntran )
    f.read_ints("int32")
    # 6  WRITE ( unit ) ( ( translation ( j, i ), j = 1, nd ), i = 1, ntran )
    f.read_reals("double")
    # 7  WRITE ( unit ) ( ( tau ( j, i ), j = 1, nd ), atomic_number ( atm ( ityp ( i ) ) ), i = 1, nat )
    read_and_throw = f.read_record(
        np.dtype(("<f8", (3))),
        np.dtype(("<i4")),
        np.dtype(("<f8", (3))),
        np.dtype(("<i4")),
    )
    # print(temp)
    # 8  WRITE ( unit ) nrecord
    f.read_ints("int32")
    # 9  WRITE ( unit ) ng_g
    f.read_ints("int32")
    # 10 WRITE ( unit ) ( ( g_g ( id, ig ), id = 1, nd ), ig = 1, ng_g )
    gvecs = f.read_ints("int32").reshape(-1, 3)

    # 11 WRITE ( unit ) nrecord
    f.read_ints("int32")
    # 12 WRITE ( unit ) ng_g
    f.read_ints("int32")
    # 13 IF ( real_or_complex .EQ. 1 ) THEN
    #     WRITE ( unit ) ( ( dble ( vxcg_g ( ig, is ) ), &
    #     ig = 1, ng_g ), is = 1, ns )
    rho = f.read_reals("double").reshape(-1, 2) @ np.array([1, 1j])
    # print(rho[:10])
    rhotuple = namedtuple("RHO", ["rho", "gvecs"])(rho, gvecs)

    f.close()

    return rhotuple


def read_vxc(filename):
    """Read Vxc data from ``vxc.dat``
    Return as a namedtuple containing vxc and kpts"""
    kpts = []
    vxc = []
    with open(filename) as f:
        while True:
            kpt = f.readline().strip().split()
            if not kpt:
                break
            kpt[:3] = map(float, kpt[:3])  # kpt vec
            kpt[3] = int(kpt[3])  # nbands
            kpts.append(kpt)

            vxc_dat = []
            for _ in range(kpt[3]):
                vxc_dat.append(float(f.readline().strip().split()[2]))
            vxc.append(vxc_dat)

    vxctuple = namedtuple("VXC", ["vxc", "kpts"])(vxc, kpts)

    return vxctuple


def read_epsilon_inp(filename="./QE_data/control_scripts/epsilon.inp"):
    """Epsilon input reader
    Reads the ``epsilon.inp`` file via: JSON -> Dict -> Namedtuple
    Use epsdata.__doc__ to see contents.

    Returns
    -------
    epsdata: NamedTuple object
    """

    def read_input_epsilon_dict(filename) -> Dict:
        """Load ``epsilon.inp`` file data to dictionary,
        doing automatic type inference using JSON module"""
        
        with open(filename, "r") as file:
            jsonstr = "{  " # Beginning of JSON string
            qpts = []       # List of qpoints
            is_q0 = []      # List of bools: "is qpt 0?"
            options = []    # List of options

            for line in file:
                linedata = line.strip().split()

                if len(linedata) == 0:
                    continue

                elif linedata[0] == "begin" and linedata[1] == "qpoints":
                    # Read and parse next line
                    line = next(file)
                    linedata = line.strip().split()
                    # Begin reading q-points
                    while linedata[0] != "end":
                        qpts.append(list(map(float, linedata[:4])))
                        is_q0.append(bool(int(linedata[4])))
                        # Data format: qx qy qz 1/scale_factor is_q0

                        line = next(file)
                        linedata = line.strip().split()

                elif len(linedata) > 1:
                    # Add data to JSON string
                    jsonstr += '"' + linedata[0] + '":' + linedata[1] + ","

                elif len(linedata) == 1:
                    options.append(linedata[0])

        jsonstr = jsonstr[:-1]  # Remove trailing comma
        jsonstr += "}"          # Close JSON string
        
        # Create epsilon input data dictionary by converting from the above jsonstr
        eps_inp_dict = json.loads(jsonstr, parse_float=float, parse_int=int)
        # Add specially parsed kpoints and options data to sigma_inp_dict
        eps_inp_dict["qpts"] = np.array(qpts)
        eps_inp_dict["is_q0"] = is_q0
        eps_inp_dict["options"] = options
        return eps_inp_dict
    
    tempdict = read_input_epsilon_dict(filename)
    epsdata = dict_to_namedtuple(tempdict, "EpsilonInp")
    return epsdata



def read_sigma_inp(filename="./QE_data/control_scripts/sigma.inp"):
    """Sigma.inp reader
    Reads the ``sigma.inp`` file via: JSON -> Dict -> Namedtuple

    Use sigmadata.__doc__ to see contents

    Returns
    ----------
    sigmadata: NamedTuple object
    """

    sigmadata: NamedTuple

    def read_input_sigma_dict(filename):
        """Load ``sigma.inp`` file data to dictionary,
        doing automatic type inference using JSON module"""

        with open(filename, "r") as file:

            # Beginning of JSON string
            jsonstr = "{  " 

            kpts = []       # List of kpoints
            options = []    # List of options

            for line in file:
                linedata = line.strip().split()

                if len(linedata) == 0:
                    continue

                elif linedata[0] == "begin" and linedata[1] == "kpoints":
                    # Read and parse next line
                    line = next(file)
                    linedata = line.strip().split()
                    # Read k-points
                    while linedata[0] != "end":
                        kpts.append(list(map(float, linedata[:4])))
                        #  kx  ky  kz  1/scale_factor
                        line = next(file)
                        linedata = line.strip().split()

                elif len(linedata) > 1:
                    # Add data to JSON string
                    jsonstr += '"' + linedata[0] + '":' + linedata[1] + ","

                elif len(linedata) == 1:
                    options.append(linedata[0])

            jsonstr = jsonstr[:-1]  # Remove trailing comma
            jsonstr += "}"          # Close JSON string
            
            sigma_inp_dict = json.loads(jsonstr, parse_float=float, parse_int=int)
            # Add specially parsed kpoints and options data to sigma_inp_dict
            sigma_inp_dict["kpts"] = np.array(kpts)
            sigma_inp_dict["options"] = options

        return sigma_inp_dict


    sigma_inp_dict = read_input_sigma_dict(filename)
    sigmadata = dict_to_namedtuple(sigma_inp_dict, "SigmaInp")
    return sigmadata

# Classes ---------------------------------------------------------------
# Not used now, they were intermediate consideration.

# class EpsilonInp:
#     """Epsilon input class
#     Reads the ``epsilon.inp`` file via:
#     JSON -> Dict -> Namedtuple

#     Use epsdata.__doc__ to see contents.
    

#     Attributes
#     ----------
#     epsdata: NamedTuple
#     """

#     epsdata: NamedTuple

#     def __init__(self, filename="./QE_data/control_scripts/epsilon.inp") -> None:
#         tempdict = self.read_input_epsilon_dict(filename)
#         self.epsdata = dict_to_namedtuple(tempdict, "epsilon_inp")

#     def __new__(cls):
#         return cls.epsdata

#     def read_input_epsilon_dict(self, filename) -> Dict:
#         """Load ``epsilon.inp`` file data to dictionary,

#         doing automatic type inference using JSON module"""
        
#         with open(filename, "r") as file:
#             jsonstr = "{  " # Beginning of JSON string
#             qpts = []       # List of qpoints
#             is_q0 = []      # List of bools: "is qpt 0?"
#             options = []    # List of options

#             for line in file:
#                 linedata = line.strip().split()

#                 if len(linedata) == 0:
#                     continue

#                 elif linedata[0] == "begin" and linedata[1] == "qpoints":
#                     # Read and parse next line
#                     line = next(file)
#                     linedata = line.strip().split()
#                     # Begin reading q-points
#                     while linedata[0] != "end":
#                         qpts.append(list(map(float, linedata[:4])))
#                         is_q0.append(bool(linedata[4]))
#                         # Data format: qx qy qz 1/scale_factor is_q0

#                         line = next(file)
#                         linedata = line.strip().split()

#                 elif len(linedata) > 1:
#                     # Add data to JSON string
#                     jsonstr += '"' + linedata[0] + '":' + linedata[1] + ","

#                 elif len(linedata) == 1:
#                     options.append(linedata[0])

#         jsonstr = jsonstr[:-1]  # Remove trailing comma
#         jsonstr += "}"          # Close JSON string
        
#         # Create epsilon input data dictionary by converting from the above jsonstr
#         eps_inp_dict = json.loads(jsonstr, parse_float=float, parse_int=int)
#         # Add specially parsed kpoints and options data to sigma_inp_dict
#         eps_inp_dict["qpts"] = np.array(qpts)
#         eps_inp_dict["is_q0"] = is_q0
#         eps_inp_dict["options"] = options

#         return eps_inp_dict


# class SigmaInp:
#     """Sigma input class
#     Reads the ``sigma.inp`` file via:
#     JSON -> Dict -> Namedtuple

#     Use epsdata.__doc__ to see contents

#     Attributes
#     ----------
#     sigmadata: NamedTuple
#     """

#     sigmadata: NamedTuple

#     def __init__(self, filename="./QE_data/control_scripts/sigma.inp") -> None:
#         sigma_inp_dict = self.read_input_sigma_dict(filename)
#         self.sigmadata = dict_to_namedtuple(sigma_inp_dict, "sigma_inp")

#     def read_input_sigma_dict(self, filename):
#         """Load ``sigma.inp`` file data to dictionary,
#         doing automatic type inference using JSON module"""

#         with open(filename, "r") as file:

#             # Beginning of JSON string
#             jsonstr = "{  " 

#             kpts = []       # List of kpoints
#             options = []    # List of options

#             for line in file:
#                 linedata = line.strip().split()

#                 if len(linedata) == 0:
#                     continue

#                 elif linedata[0] == "begin" and linedata[1] == "kpoints":
#                     # Read and parse next line
#                     line = next(file)
#                     linedata = line.strip().split()
#                     # Read k-points
#                     while linedata[0] != "end":
#                         kpts.append(list(map(float, linedata[:4])))
#                         #  kx  ky  kz  1/scale_factor
#                         line = next(file)
#                         linedata = line.strip().split()

#                 elif len(linedata) > 1:
#                     # Add data to JSON string
#                     jsonstr += '"' + linedata[0] + '":' + linedata[1] + ","

#                 elif len(linedata) == 1:
#                     options.append(linedata[0])

#             jsonstr = jsonstr[:-1]  # Remove trailing comma
#             jsonstr += "}"          # Close JSON string
            
#             sigma_inp_dict = json.loads(jsonstr, parse_float=float, parse_int=int)
#             # Add specially parsed kpoints and options data to sigma_inp_dict
#             sigma_inp_dict["kpts"] = np.array(kpts)
#             sigma_inp_dict["options"] = options

#         return sigma_inp_dict
