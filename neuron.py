#!/usr/bin/python3

import numpy as np
import time

# input   out
# 0 0 1     0
# 1 1 1     1
# 1 0 1     1
# 0 1 1     0

dataset = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])


expected = np.array([[0, 0, 1, 1]]).T

# deterministe
np.random.seed(4)

# 3 poids aléatoires entre -1 et 1
w0 = 2*np.random.random((3,1)) - 1

print('Poids initiaux : \n {}'.format(w0.T))

def sigmoid(x, deriv):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

l0 = dataset

t0 = time.time()

for iter in range(0, 10000):

    global w0

    l0 = dataset

    # ponderation
    l1 = sigmoid(np.dot(l0, w0), False)

    # correction
    l1_error = expected - l1

    #if (iter % 1000 == 0):
    #    print('Erreur : {} \n'.format(l1_error))

    # Màj des poids
    l1_delta = l1_error * sigmoid(l1, True)
    w0 += np.dot(l0.T, l1_delta)
    #print('Poids : \n{}'.format(w0.T))

print('Poids finaux : \n {}'.format(w0.T))

t1 = time.time()

print('Temps : {} \n'.format(t1 - t0))





