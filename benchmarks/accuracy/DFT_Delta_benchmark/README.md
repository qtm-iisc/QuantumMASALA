# Delta Benchmark for Accuracy of DFT

Please refer to https://molmod.ugent.be/deltacodesdft for more information on the Delta benchmark for DFT, and for the results of the benchmark for other codes.

The results of the benchmark as discussed in the paper, are available in the `Results` directory.

To assemble the results from the SCF runs for all the elements and to calculate the Delta values with QuantumESPRESSO, run the following command:

```
rm */*.txt ; python script_create_txt_from_log.py && python calculate_eosout.py && python script_assemble_delta.py > qtm_delta.txt && python calcDelta.py qtm_delta.txt ./Results/eos-QE+SG15v1.2.txt
```