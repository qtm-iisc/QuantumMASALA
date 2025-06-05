## Run the process on 12, 15, 20, and 25 cores in parallel
for i in {1..10} 12 15 20 25 30 40
do
    # Create a directory for the current number of processors
    mkdir -p si_$i

    for j in {1..10}
    do
        echo "Running on $i cores, iteration $j"
        nohup mpirun -np $i python si_scf_supercell.py 3 > si_$i/output_${i}_$j.txt &
        wait
    done
done