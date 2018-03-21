from multiprocessing import Pool
import os
import numpy as np
import utils.data as data
import h5py

def simulate(paramFileName, outfileName, exploration_prob = 0.1, numGames=20000, poolsize = 10):
    p = Pool(poolsize)
    gamesPerIter = int(numGames/100)
    str1 = "python zsyMultigameHelper.py " + paramFileName + " temp"
    str2 = ".h5 " + str(gamesPerIter) + " " + str(exploration_prob)
    commands = [str1 + str(i) + str2 for i in range(100)]
    ret = p.map(os.system, commands)
#     print(ret)
    outfiles = ["temp"+str(i)+".h5" for i in range(100)]
    DATA = [data.dataFileToLabeledData_1(outfile) for outfile in outfiles]
    X_A = np.concatenate([d[0] for d in DATA], axis=1)
    X_B = np.concatenate([d[1] for d in DATA], axis=1)
    Y_A = np.concatenate([d[2] for d in DATA], axis=1)
    Y_B = np.concatenate([d[3] for d in DATA], axis=1)
    print("\n X_A shape:"+str(X_A.shape))
    f = h5py.File(outfileName, "w")
    XAset = f.create_dataset("X_A", X_A.shape, compression="gzip")
    XAset[...] = X_A
    XBset = f.create_dataset("X_B", X_B.shape, compression="gzip")
    XBset[...] = X_B
    YAset = f.create_dataset("Y_A", Y_A.shape, compression="gzip")
    YAset[...] = Y_A
    YBset = f.create_dataset("Y_B", Y_B.shape, compression="gzip")
    YBset[...] = Y_B
    f.close()
    [os.remove(outfile) for outfile in outfiles]

