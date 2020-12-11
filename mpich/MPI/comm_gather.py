# comm_gather.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = rank
print('before gathering: process %d has %s' % (rank, data))

data = comm.gather(data, root=0)
print('after scattering: process %d has %s' % (rank, data))