from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score
from os import fork
from os import kill 

from mpi4py import MPI

import numpy as np 
import datetime as dt
import time 
import csv 
#psutil: process and system utilities
import psutil
import signal

def cpu(file_name, rank, size):
    cpu_file = open(file_name+str(rank)+"_cpu.txt", "a")
    #cpu_file.write('#Number of processes : %d\n'%(size))
    while 1:
        # cpu used percent
        cpu_percent = psutil.cpu_percent()
        #virtual memory info: total, available, percent, used, free, active
        #memory_percent = psutil.virtual_memory()[2]
        #disk usage info: total, used, free, percent
        #disk_percent = psutil.disk_usage('/')[3]
        # network interface package counts (return a dict with pernic = True)
        #net_io_recv = psutil.net_io_counters(pernic=True)['packets_recv']
        #net_io_sent = psutil.net_io_counters(pernic=True)['packets_sent']
        time.sleep(0.001) 
        cpu_file.write('%d,%2.2f\n' % (size, cpu_percent))      
        

def MPI_temp():
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()
    size = comm.Get_size()
    # for node 1-8, 1680!
    L = 84000 # length before scatter
    l = int(L / size) # length after scatter
    # on all nodes, create variables for later use
    X_train = None
    Y_train = None
    Pred = None
    
    done = 0

    file_name = "./sample"+str(L)+"_mpi_sklearn_rf_" 
    
    if rank == 0:       
        # Load data from a text file, specify seperator as ','
        train = np.genfromtxt('temptrain.csv', delimiter=',')
        ## ?? train1 = train.reshape(157774)
        samples = 100000
        sample1 = train[1:samples]
    
        d1 = sample1[0:samples-3]
        d2 = sample1[1:samples-2]
        d3 = sample1[2:samples-1]
        # .reshape(-1, 1) change shape to one column(do not specify row)
        # np.concatenate((arr1, arr2,...), axis=1): append arrays by row
        X_train = np.concatenate((d1.reshape(-1,1),
                                  d2.reshape(-1,1),
                                  d3.reshape(-1,1)),axis=1)
        Y_train = sample1[2:samples]
        #Y_train = Y_train.T
        print("\n"+str(size)+"\n\n")
        X_train = X_train[:L,:]  
        Y_train = Y_train[:L] 
        Pred = np.zeros([l, size])
        
    # define received data on each process(according to size)
    part_x = np.zeros([l,3])
    part_y = np.zeros(l) 
    
    # fork() copy a process
    newpid = fork()
    # only on child process(fork() returns 0) of rank 0, call cpu()
    if newpid == 0:
        cpu(file_name, rank, size)
    # on parent process(fork() returns new ID of process)
    else:
        t_start = MPI.Wtime()
        comm.Scatter(X_train, part_x, root = 0) 
        comm.Scatter(Y_train, part_y, root = 0)       
        clf = RandomForestClassifier(max_depth= 6, min_samples_leaf=9,
                                     n_estimators = 50,
                                     min_samples_split=15,
                                     max_features=0.6, oob_score=True)
        clf.fit(part_x, part_y) 
        pred = clf.predict(part_x) 
        comm.Gather(pred, Pred, root=0)
        t_end = MPI.Wtime()
        latency = t_end - t_start
        # record latency on each node
        file = open(file_name+"latency.txt", "a") 
        file.write('%d,%d,%.4f\n' % (size, rank, latency))

        if rank == 0:
            score = f1_score(Y_train, Pred.reshape(L), average='micro')
            #open(file, mode='a'): open for writing, appending
            file = open(file_name+"score.txt", "a") 
            file.write('%d, %f\n' % (size, score))
        time.sleep(5)
        kill(newpid, signal.SIGKILL)
        
        
if __name__ == '__main__':
    MPI_temp()      
