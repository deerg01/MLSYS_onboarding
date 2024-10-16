import numpy as np

def relu(x):
    return np.maximum(0, x)

def init_network():
    network = {}
    network['W1'] = np.random.rand(2, 3)
    network['b1'] = np.ones(3,)
    network['W2'] = np.random.rand(3, 2)
    network['b2'] = np.ones(2,)
    network['W3'] = np.random.rand(2,1)
    network['b3'] = np.ones(1,)
    return network

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.matmul(x, w1) + b1
    a2 = np.matmul(a1, w2) + b2
    a3 = np.matmul(a2, w3) + b3

    out = relu(a3)
    return out

network = init_network()
x = np.array([[1, 2], [3, 4]])
output = forward(network, x)

print(output, output.shape)
