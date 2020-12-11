# comm_bcast_numpy.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import numpy as np

if rank == 0:
    data = np.arange(10, dtype='i')
    print('before bcast: process %d has %s' %(rank, data))
else:
    data = np.zeros(10, dtype='i')
    print('before bcast: process %d has %s' %(rank, data))

comm.Bcast(data, root=0)
print('after bcast: process %d has %s' %(rank, data))