from mpi4py import MPI

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

print("hello world! I'm processor %d of %d on %s \n" % (rank, size, name))

