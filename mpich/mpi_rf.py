from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score

from mpi4py import MPI
import numpy as np 

def getSize():
    iris = datasets.load_iris() 
    return len(iris.data)  

def MPI_rf():
    # mpi4py: MPI.COMM_WORLD/.Get_rank()/.Get_size()
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()
    size = comm.Get_size()
    # for node 1-8, 1680!
    L = 16800 # length before scatter
    l = int(L / size) # length after scatter
    # on all nodes, create variables for later use
    X_train = None
    Y_train = None
    Pred = None
    # for recording
    file_name = "./sample"+str(L)+"_mpi_sklearn_rf_"    
    # on master, import/transform datas
    if rank == 0:
        # Load data from a text file, specify seperator as ','
        train = np.genfromtxt('temptrain.csv', delimiter=',')
        ## ?? train1 = train.reshape(157774)
        # subset a sample
        samples = 20000
        sample = train[1:samples]
        
        d1 = sample[0:samples-3]
        d2 = sample[1:samples-2]
        d3 = sample[2:samples-1]
        # .reshape(-1, 1) change shape to one column(do not specify row)
        # np.concatenate((arr1, arr2,...), axis=1): append arrays by row
        X_train = np.concatenate((d1.reshape(-1,1),
                                  d2.reshape(-1,1),
                                  d3.reshape(-1,1)),axis=1)
        Y_train = sample[2:samples]
        #Y_train = Y_train.T
        
        #confirm number of nodes
        print("\n"+str(size)+"\n\n")
        #data to be scattered
        X_train = X_train[:L,:]  
        Y_train = Y_train[:L] 
        # create to gather
        Pred = np.zeros([l, size])
        
    # create to receive data on each processl
    part_x = np.zeros([l,3])
    part_y = np.zeros(l) 
    
    # timer for communication and modelling
    t_start = MPI.Wtime()
    # scatter to each process
    comm.Scatter(X_train, part_x, root = 0) 
    comm.Scatter(Y_train, part_y, root = 0) 
    #define fit model
    rfc = RandomForestClassifier(max_depth= 6, min_samples_leaf=9, 
                                 n_estimators = 50, 
                                 min_samples_split=15, 
                                 max_features=0.6, oob_score=True)
    rfc.fit(part_x, part_y) 
    pred = rfc.predict(part_x)
    # gather pred to process 0
    comm.Gather(pred, Pred, root=0)
    # modeling done
    t_end = MPI.Wtime()
    latency = t_end - t_start
    
    # record latency on each node
    file = open(file_name+"latency.txt", "a") 
    file.write('%d,%d,%.4f\n' % (size, rank, latency))
    # Evaluation
    if rank == 0 :
        score = f1_score(Y_train, Pred.reshape(L), average='micro')
        # open(file, mode='a'): open for writing, appending
        file = open(file_name+"score.txt", "a") 
        file.write('%d,%f\n' % (size, score))
# if this file is executed directly, run MPI_rf()                
if __name__ == '__main__':
    MPI_rf()      