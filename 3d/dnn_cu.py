import cupy as cp
from keras.datasets import mnist

class nn:
    def __init__(self, ip, hid, op):
        self.ip = ip
        self.hid = hid
        self.op = op
        
        # Initialize weights and biases
        self.w_iphid = cp.random.randn(ip, hid)
        self.w_hidop = cp.random.randn(hid, op)
        self.b_iphid = cp.zeros((1, hid))
        self.b_hidop = cp.zeros((1, op))
    
    def forward(self, X):
        # Forward pass through the network
        self.h_ip = cp.dot(X, self.w_iphid) + self.b_iphid
        self.h_op = cp.maximum(0, self.h_ip)  
        self.output = softmax(cp.dot(self.h_op, self.w_hidop) + self.b_hidop) 
        return self.output
    
    def backprop(self, X, y, lrate):
        # Backpropagation
        # Compute gradients
        err = self.output - y
        op_diff = err
        hid_diff = cp.dot(op_diff, self.w_hidop.T) * (self.h_op > 0)  
        
        # Update weights and biases
        self.w_hidop -= lrate * cp.dot(self.h_op.T, op_diff)
        self.b_hidop -= lrate * cp.sum(op_diff, axis=0, keepdims=True)
        self.w_iphid -= lrate * cp.dot(X.T, hid_diff)
        self.b_iphid -= lrate * cp.sum(hid_diff, axis=0, keepdims=True)
    
    def train(self, X, y, lrate, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backprop(X, y, lrate)
            if epoch % 10 == 0:
                loss = cp.mean(cp.square(y - output))
                print(f'epoch {epoch}, loss: {loss}')
    
    def predict(self, X):
        return self.forward(X)

# Activation function
def softmax(x):
    exp = cp.exp(x - cp.max(x, axis = 1, keepdims = True))
    out = exp / cp.sum(exp, axis = 1, keepdims = True)
    return out

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape images to 1D arrays
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    # One-hot encode labels
    y_train = cp.eye(10)[y_train]
    y_test = cp.eye(10)[y_test]
    return X_train, y_train, X_test, y_test


ip = 784  
hid = 74
op = 10  
lrate = 0.001
epochs = 100

x, y, x_, y_ = load_data()
dnn = nn(ip, hid, op)
dnn.train(x, y, lrate, epochs)

predict = dnn.predict(x_)
acc = cp.mean(cp.argmax(predict, axis=1) == cp.argmax(y_, axis=1))
print(f'accuracy: {acc.get()}')
