from utils.runn import *
import time
import os
import zsyGame as zsy
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from datetime import timedelta
import utils.data as data
import math

def saveHP(model):
    if not os.path.isdir(model.home):
        os.mkdir(model.home)
    with open(os.path.join(model.home, 'hyperparams_' + str(model.modelIter)) + '.pkl', 'wb+') as f:
                pickle.dump(zsyDenseHPSummary(model), f)

def loadHP(hpFileName):
    with open(hpFileName, 'rb') as f:
        hpInstance = pickle.load(f)
    return Model(hpInstance.layers, hpInstance.dataFilename, hpInstance.home, hpInstance.keep_prob, hpInstance.c_discount)

class zsyDenseHPSummary:
    def __init__(self, hpInstance):
        self.layers = hpInstance.layers
        self.dataFilename = hpInstance.dataFilename
        self.keep_prob = hpInstance.keep_prob
        self.r_discount = hpInstance.r_discount
        self.c_discount = hpInstance.c_discount
        self.home = hpInstance.home


class Model:
    def __init__(self, layers,  dest_folder, dataFilename = 'data/1M RandvRand/T100k_1.h5', keep_prob=0.5, r_discount=1, c_discount=1):
        self.X = tf.placeholder(tf.float32, [layers[0], None])
        self.Y = tf.placeholder(tf.float32, [2, None])
        self.layers = layers
        self.dataFilename = dataFilename
        self.keep_prob = keep_prob
        self.r_discount = r_discount
        self.c_discount = c_discount
        self.params = {}
        self.home = os.path.abspath(dest_folder)
        self.nodes = {'A0': self.X, 'A_0': self.X} # underscore for w/o dropout
        for i in range(1, len(layers)):
            # Nodes with '_' in middle indicate no dropout
            self.params['W' + str(i)] = tf.get_variable('W' + str(i), [layers[i], layers[i-1]],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            self.params['b' + str(i)] = tf.get_variable('b' + str(i), [layers[i],1],
                                                   initializer=tf.zeros_initializer)
            self.nodes['Z' + str(i)] = tf.add(tf.matmul(self.params['W'+str(i)], self.nodes['A'+str(i-1)]), self.params['b'+str(i)],
                                         name='Z' + str(i))
            self.nodes['Z_' + str(i)] = tf.add(tf.matmul(self.params['W' + str(i)], self.nodes['A_' + str(i-1)]), self.params['b' + str(i)],
                                         name='Z_' + str(i))
            if i != len(layers)-1:
                self.nodes['A' + str(i)] = tf.nn.dropout(tf.nn.relu(self.nodes['Z'+str(i)]), keep_prob,
                                                    name='A' + str(i))
                self.nodes['A_' + str(i)] = tf.nn.relu(self.nodes['Z_' + str(i)], name='A_' + str(i))
            else:
                self.nodes['A' + str(i)] = tf.sigmoid(self.nodes['Z'+str(i)], name='A' + str(i))
                self.nodes['A_' + str(i)] = tf.sigmoid(self.nodes['Z_' + str(i)], name='A_' + str(i))
        self.lastNodeName = "A" + str(len(layers)-1)
        self.lastNodeName_ = "A_" + str(len(layers)-1)
        self.nodes['cost'] = tf.losses.log_loss(tf.transpose(self.nodes[self.lastNodeName]),
                                                tf.transpose(self.Y[0:1]), weights=tf.transpose(self.Y[1:2]))
        self.nodes['cost_'] = tf.losses.log_loss(tf.transpose(self.nodes[self.lastNodeName_]),
                                                tf.transpose(self.Y[0:1]), weights=tf.transpose(self.Y[1:2]))
        self.costs = []
        self.dev_costs = []
        self.total_epochs = 0
        self.modelIter = 1
        while True:
            if os.path.isfile(os.path.join(self.home, 'hyperparams_' + str(self.modelIter))):
                self.modelIter += 1
            else:
                break
        self.paramFileName = os.path.join(self.home, 'params_'+os.path.basename(self.home)+'_'+str(self.modelIter)+'.pkl')

    def setData(self):
        X_A, X_B, Y_A, Y_B = data.dataFileToLabeledData_1(self.dataFilename)
        self.X_Train = np.concatenate([X_A[:, :int(X_A.shape[1]*.98)], X_B[:, :int(X_B.shape[1]*.98)]], axis=1)
        self.X_Dev = np.concatenate([X_A[:, int(X_A.shape[1]*.98):], X_B[:, int(X_B.shape[1]*.98):]], axis=1)
        self.Y_Train = np.concatenate([Y_A[:, :int(X_A.shape[1]*.98)], Y_B[:, :int(X_B.shape[1]*.98)]], axis=1)
        self.Y_Dev = np.concatenate([Y_A[:, int(X_A.shape[1]*.98):], Y_B[:, int(X_B.shape[1]*.98):]], axis=1)

        self.Y_Train[0] = (self.Y_Train[0]*self.r_discount**self.Y_Train[1] + 1)/2 # Discounted reward
        self.Y_Train[1] = self.c_discount**self.Y_Train[1] # Discounted cost
        self.Y_Dev[0] = (self.Y_Dev[0]*self.r_discount**self.Y_Dev[1] + 1)/2
        self.Y_Dev[1] = self.c_discount**self.Y_Dev[1] # Discounted cost

    def TrainModel(self, num_epochs=100, learning_rate=0.001, minibatch_size=1024, print_cost=True):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.nodes['cost'])
        init = tf.global_variables_initializer()
        (n_x, m) = self.X_Train.shape
        with tf.Session() as sess:
            # Run the initialization
            sess.run(init)
            # Do the training loop
            for epoch in range(num_epochs):
                if epoch == 0:
                    tic0 = time.time()
                if epoch == 10:
                    tic10 = time.time()
                epoch_cost = 0.  # Defines a cost related to an epoch
                num_minibatches = int( m / minibatch_size)
                if epoch == 0:
                    print("num minibatches:%d" % num_minibatches)
#                 minibatches = random_mini_batches(self.X_Train, self.Y_Train, minibatch_size, 3)
                tic = time.time()
                permutation = list(np.random.permutation(m))
                num_minibatches = math.ceil(m/minibatch_size)
                for k in range(0, num_minibatches):
                    _, minibatch_cost = sess.run([optimizer, self.nodes['cost']], feed_dict={
                        self.X: self.X_Train[:,permutation[k * minibatch_size : min((k+1) * minibatch_size, m)]],
                        self.Y: self.Y_Train[:,permutation[k * minibatch_size : min((k+1) * minibatch_size, m)]]})
                    epoch_cost += minibatch_cost / num_minibatches
                toc = time.time()

                if epoch == 0:
                    print("\tEpoch time:%d seconds" % (toc - tic))
                if epoch == 10:
                    print("\tAve epoch time: %f" % ((tic10 - tic0) / 10))
                    print("\t100 epochs will take %s" % (timedelta(seconds=10 * (tic10 - tic0))))
                if print_cost and (epoch % 5 == 0 or epoch < 10):
                    dev_cost = sess.run(self.nodes['cost_'], feed_dict={self.X: self.X_Dev, self.Y: self.Y_Dev})
                if print_cost and (epoch % 10 == 0 or epoch < 10):
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                    print("\tDev error (no dropout): %f" % dev_cost)
                if print_cost and epoch % 5 == 0:
                    self.costs.append(epoch_cost)
                    self.dev_costs.append(dev_cost)

            xrange = 5 * np.arange(num_epochs / 5)
            plt.plot(xrange, np.squeeze(self.costs), xrange, np.squeeze(self.dev_costs))
            plt.ylabel('cost')
            plt.xlabel('iterations')
            plt.title("Learning rate =" + str(learning_rate))
            plt.savefig(os.path.join(self.home, 'costs_'+str(self.modelIter)+'.png'))

            parameters = sess.run(self.params)
            with open(self.paramFileName, 'wb+') as f:
                pickle.dump(parameters, f)
            print("Parameters have been trained!")
            train_cost = sess.run(self.nodes['cost_'], feed_dict={self.X: self.X_Train, self.Y: self.Y_Train})
            dev_cost = sess.run(self.nodes['cost_'], feed_dict={self.X: self.X_Dev, self.Y: self.Y_Dev})
            print("Train cost (No dropout):", train_cost)
            print("Test cost (No dropout):", dev_cost)

    def TestModel(self):
        zsy.stdTest(self.paramFileName)

    def Whole(self):
        self.setData()
        self.TrainModel()
        self.TestModel()




