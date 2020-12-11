# comm_gather_numpy.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sdata = np.zeros(3) + rank
print('before gathering: process %d has %s' %(rank, sdata))

rdata = None
if rank == 0:
    rdata = np.zeros([size, 3])

comm.Gather(sdata, rdata, root=0)
print('after gathering, process %d has\n%s' %(rank, rdata))