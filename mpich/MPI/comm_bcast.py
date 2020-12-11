# comm_bcast.py
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = 1
    print('before bcast: process %d has %s' %(rank, data))
else:
    data = 0
    print('before bcast: process %d has %s' %(rank, data))

data = comm.bcast(data, root=0)
print('after bcast: process %d has %s' %(rank, data))