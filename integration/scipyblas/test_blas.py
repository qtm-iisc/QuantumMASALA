import numpy as np
from scipy.linalg.blas import zgemm
from time import sleep

from argparse import ArgumentParser
argparser = ArgumentParser()
argparser.add_argument('--flag1', action='store_true')
argparser.add_argument('--flag2', action='store_true')
args, _ = argparser.parse_known_args()

def create_rand_c16(shape):
    return np.random.rand(*shape) + 1j*np.random.rand(*shape)

bra_shape = (2, 3, 40)
ket_shape = (5, 60)
numgk = 25000

bra = create_rand_c16((*bra_shape, numgk))
ket = create_rand_c16((*ket_shape, numgk))
braket
sleep(0.5)

a = bra.reshape((-1, numgk))
b = ket.reshape((-1, numgk))

if args.flag1:
    print('flag 1 passed')
    a = a.T
    b = b.T

    print(a.shape, b.shape)
    braket = zgemm(
        alpha=1.0, a=a, trans_a=2,
        b=b, trans_b=0,
    )
    print(braket.shape)
    print(braket.flags)
    # braket = braket.reshape((*bra_shape, *ket_shape))
else:
    print('flag 1 NOT passed')
    # a, b = b, a
    # print(a.shape, b.shape)
    # braket = zgemm(
    #     alpha=1.0, a=a, trans_a=0,
    #     b=b, trans_b=2,
    # ).T
    
    braket = a.conj() @ b.T
    print(braket.shape)
    
print('done')
sleep(0.5)

if args.flag2:
    print('flag 2 passed')
    test = bra.reshape((-1, numgk)).conj() @ ket.reshape((-1, numgk)).T
    print(np.allclose(test, braket))

sleep(0.5)
