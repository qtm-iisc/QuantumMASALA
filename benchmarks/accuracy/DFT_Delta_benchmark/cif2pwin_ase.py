import os
import shutil

import csv

import spglib
from ase.io.cif import read_cif
from ase.io.espresso import write_espresso_in

from spglib import *
import numpy as np

small_calc = False

cif_dir = './CIFs/'
qe_in_dir = './SCFin/'
test_dir = './SCFtest/'

if os.path.isdir(qe_in_dir):
    shutil.rmtree(qe_in_dir)
os.mkdir(qe_in_dir)
if os.path.isdir(test_dir):
    shutil.rmtree(test_dir)
os.mkdir(test_dir)

pseudo_dir = os.path.join(qe_in_dir, 'pseudo/')

os.mkdir(pseudo_dir)
pseudotar = 'sg15_oncv_upf_2020-02-06.tar.gz'
upf_suffix = '_ONCV_PBE-1.2.upf'

os.system("tar -xzf " + pseudotar + ' --directory=' + pseudo_dir)

control = {'calculation': 'scf',
           'prefix': 'scf',
           'pseudo_dir': './',
           'verbosity': 'high',
           'disk_io':'nowf'
           }

system = {'ecutwfc': 100, 'nspin': 1,
          'occupations': 'smearing', 'smearing': 'gaussian', 'degauss': 0.0007349862,
          }

electrons = {'conv_thr': 1E-10,
             }  # Check if this tolerance is good enough

input_data = {**control, **system, **electrons,
              }


elemlist_fname = 'elemlist.csv'
elem_table = []
with open('elemlist.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        elem_table.append([int(row[0]), row[1], int(row[2]),
                           [int(row[3+i]) for i in range(3)]])

l_scale = [94, 96, 98, 100, 102, 104, 106]
l_ferro = ['Fe', 'Co', 'Ni']
l_antiferro = ['O', 'Cr', 'Mn']


for elem in elem_table:
    elem_num = elem[0]
    elem_name = elem[1]
    val = elem[2]
    kpts_grid = tuple(elem[3])
    kpts_grid_test = tuple([n//4 for n in kpts_grid])
    kpts_shift = (1, 1, 1)

    cif_file = os.path.join(cif_dir, elem_name + '.cif')
    cryst = list(read_cif(cif_file, slice(None), ))
    #print(len(cryst))
    cryst = cryst[0]

    elem_dir = os.path.join(qe_in_dir, str(elem_num) + '-' + elem_name)
    os.mkdir(elem_dir)
    test_elem_dir = os.path.join(test_dir, str(elem_num) + '-' + elem_name)
    os.mkdir(test_elem_dir)

    upf_name = elem_name + upf_suffix
    if not os.path.isfile(os.path.join(pseudo_dir, upf_name)):
        print("UPF file missing in pseudo directory")
        raise ValueError("Missing UPF File")
    shutil.copy(os.path.join(pseudo_dir, upf_name), elem_dir)
    shutil.copy(os.path.join(pseudo_dir, upf_name), test_elem_dir)

    num_at = len(cryst)
    if elem_name in l_ferro:
        system['nspin'] = 2
        cryst.set_initial_magnetic_moments([+1 for i_at in range(num_at)])
    elif elem_name in l_antiferro:
        system['nspin'] = 2
        if elem_name == 'O':
            cryst.set_initial_magnetic_moments([(-1)**i_at for i_at in range(num_at)])
        else:
            cryst.set_initial_magnetic_moments([(-1)**i_at for i_at in range(num_at)])
    else:
        system['nspin'] = 1

    cryst.center()
    lattice = np.array(cryst.cell[:])
    alat = np.linalg.norm(lattice[0])
    #lattice /= alat
    positions = cryst.get_scaled_positions()
    atoms = cryst.get_atomic_numbers()

    cell = (lattice, positions, atoms)
    #cell = standardize_cell(cell, to_primitive=False, no_idealize=False, symprec=1e-7)
    symmetry = get_symmetry_dataset(cell, symprec=1e-7, angle_tolerance=0.0)
    #print('symm', len(symmetry['rotations']))
    mapping, grid = get_ir_reciprocal_mesh(kpts_grid, cell, kpts_shift, symprec=1e-12,
                                           is_time_reversal=False)

    pseudo = {elem_name: upf_name}
    print(elem_name, len(symmetry['rotations']), len(np.unique(mapping)) * system['nspin'],
          kpts_grid, cryst.get_initial_magnetic_moments())
    cellpar = cryst.cell.cellpar()
    #print(lattice)

    qe_in_name = elem_name + '-test' + '.scf.in'
    qe_in_file = open(os.path.join(test_elem_dir, qe_in_name), 'w')
    input_data['ecutwfc'] = 40
    input_data['prefix'] = elem_name + '-test'
    write_espresso_in(fd=qe_in_file, atoms=cryst, input_data=input_data,
                      pseudopotentials=pseudo,
                      kpts=kpts_grid_test, koffset=kpts_shift)
    qe_in_file.close()
    input_data['ecutwfc'] = 100

    for scale in l_scale:
        scale_fact = (scale/100) ** (1./3)
        cellpar_scaled = [cellpar[0] * scale_fact, cellpar[1] * scale_fact, cellpar[2] * scale_fact,
                          cellpar[3], cellpar[4], cellpar[5]]
        cryst.set_cell(cellpar_scaled, scale_atoms=True)

        qe_in_name = elem_name + '-' + str(scale) + '.scf.in'
        input_data['prefix'] = elem_name + '-' + str(scale)
        qe_in_file = open(os.path.join(elem_dir, qe_in_name), 'w')
        write_espresso_in(fd=qe_in_file, atoms=cryst, input_data=input_data,
                          pseudopotentials=pseudo,
                          kpts=kpts_grid, koffset=kpts_shift)
        qe_in_file.close()
