from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    # dict with {key: value}, set with {key}
    # Sequence: list with [], Tuple with ()
    data = {'a': 1, 'b': 2}
    print('process %d sends %s' %(rank, data))
    comm.send(data, dest=1)
elif rank == 1:
    data = comm.recv(source=0)
    print('process %d receives %s' %(rank, data))