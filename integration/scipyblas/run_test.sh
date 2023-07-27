
rm fblas.bin
memray run -o fblas.bin test_blas.py --flag1
rm fblas.html
memray flamegraph -o fblas.html fblas.bin

rm matmul.bin
memray run -o matmul.bin test_blas.py
rm matmul.html
memray flamegraph -o matmul.html matmul.bin
