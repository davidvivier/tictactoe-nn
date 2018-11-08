#!/usr/bin/python3

import numpy as np
import time

# input   out
# 0 0 1     0
# 1 1 1     1
# 1 0 1     1
# 0 1 1     0

list = [ 0, 0, 1]
#list = sys.argv

def sigmoid(x):
    return 1/(1+np.exp(-x))

#@timeit
def neuron(input):
    weight = [ 3.5, 2, -4]
    # ponderation
    sum = 0
    for i in range(0, len(input)):
        sum += input[i]*weight[i]
    result = sigmoid(sum)
    return result

print(neuron(list))


