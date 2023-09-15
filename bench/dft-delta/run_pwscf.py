import os
import subprocess
import shutil
import re
import time
import signal

# Constants
bohr_to_ang = 0.529177249
ryd_to_ev = 13.605698066

# Parameters
parallel_run = True
n_threads = 10
warn_timer = 300
kill_timer = 600

# Main Directories
in_dir = "./SCFin"
calc_dir = "./SCFcalc"
out_dir = "./SCFout"

# Cleaning calc_dir
if os.path.isdir(calc_dir):
    shutil.rmtree(calc_dir)
os.mkdir(calc_dir)
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Commands to run calculators
qe_pw_dir = "../../../../pw.x"
run_qe = [qe_pw_dir, "-in"]

pypwscf_dir = "../../../../src/pw.py"
run_pypwscf = ["python", pypwscf_dir]

if parallel_run:
    run_mpi = ["mpirun", "-n", str(n_threads)]
    run_qe = run_mpi + run_qe
    run_pypwscf = run_mpi + run_pypwscf

# Choosing which calculator
calculator = run_pypwscf

# Choosing the element range to benchmark
l_elem_dir = [elem_dir for elem_dir in os.listdir(in_dir)
              if os.path.isdir(os.path.join(in_dir, elem_dir))
              and len(re.findall('\d+', elem_dir)) == 1]
l_elem_dir.sort(key=lambda elem_dir: int(re.findall('\d+', elem_dir)[0]))

print("Directories to calculate:\n")
print(l_elem_dir)

# Running Benchmark
for elem_dir in l_elem_dir:
    # Finding element
    atom_num = int(re.findall('\d+', elem_dir)[0])
    elem_name = elem_dir.split('{:d}-'.format(atom_num), 1)[1]
    # Copying the folder corresponding to element to calc_dir
    shutil.copytree(os.path.join(in_dir, elem_dir),
                    os.path.join(calc_dir, elem_dir))

    os.chdir(calc_dir)
    print("----------Calculating Element {:}----------".format(elem_dir))
    elem_start_time = time.perf_counter()
    # Finding all input files
    l_calc = [file[:-7] for file in os.listdir(elem_dir) if file.endswith('.scf.in')]
    l_upf = [file for file in os.listdir(elem_dir) if file.endswith('.upf')]
    if len(l_upf) == 0:
        raise ValueError("UPF File Missing")

    l_vol_en = []
    error_flag = False
    for calc in l_calc:
        if os.path.isdir('calc'):
            shutil.rmtree('calc')
        os.mkdir('calc')
        calc_in = calc + '.scf.in'
        calc_out = calc + '.scf.out'
        print("-----Input File: {:}-----".format(calc_in))

        out_file = open(os.path.join('./calc', 'calc.log'), 'w+')

        # Copying input files to 'calc' folder in calc_dir
        shutil.copy(os.path.join(elem_dir, calc_in), os.path.join('calc', calc_in))
        for upf in l_upf:
            shutil.copy(os.path.join(elem_dir, upf), os.path.join('calc', upf))

        # Running calculation; Output is duplicated to stdout and *.scf.out
        run_calc = subprocess.Popen(calculator + [calc_in, ] + ['|', 'tee', calc_out],
                                    cwd='./calc/', text=True, stdout=subprocess.PIPE)
        print('PID: {:}'.format(run_calc.pid))

        # Finding cell volume and total energy from output
        energy, volume = None, None
        timer = time.perf_counter()
        while run_calc.poll() is None:
            time.sleep(1)
            if time.perf_counter() - timer > kill_timer:
                print("No output generated for the last {:d} secs. Terminating calculation"
                      .format(kill_timer))
                os.killpg(os.getpgid(run_calc.pid), signal.SIGKILL)
            for line in run_calc.stdout:
                out_file.write(line)
                timer = time.perf_counter()
                if "number of atoms/cell" in line:
                    num_at = re.findall("\d+", line)[0]
                    num_at = int(num_at)
                    print(line)
                if "unit-cell volume" in line:
                    volume = re.findall("\d+\.\d+", line)[0]
                    volume = float(volume)
                    volume *= (bohr_to_ang ** 3)
                    print(line)
                elif "!" in line and 'total energy' in line:
                    energy = re.findall("[+-]?\d+\.\d+", line)[0]
                    energy = float(energy)
                    energy *= ryd_to_ev
                    print(line)
                out_file.flush()

        out_file.close()
        # Saving to l_vol_en
        if run_calc.returncode != 0 or energy is None or volume is None:
            print("Something wrong went with the calculcation for {:}".format(calc))
            print("Moving to next calculcation. Delta factor wont be calculated")
            error_flag = True
        l_vol_en.append([volume / num_at, energy / num_at])

        # Renaming 'calc' folder to string in variable calc; e.g 'H-104'
        shutil.move('calc', calc_in[:-7])
        shutil.move(calc_in[:-7], os.path.join(elem_dir, calc_in[:-7]))

    # Move directory to out_dir

    if not error_flag:
        delta_data = open(os.path.join(elem_dir, elem_name + '.txt'), 'w')
        for vol, en in l_vol_en:
            delta_data.write("{:f} {:f}\n".format(vol, en))
        delta_data.close()
        print("Printing data for delta calculation")
        print(delta_data)

    print("Total run time for element - {:}".format(time.perf_counter() - elem_start_time))

    if os.path.isdir(os.path.join('../', out_dir, elem_dir)):
        shutil.rmtree(os.path.join('../', out_dir, elem_dir))
    shutil.move(elem_dir, os.path.join('../', out_dir, elem_dir))
    os.chdir('../')
