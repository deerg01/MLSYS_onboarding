from .loss import CrossEntropyLoss
import numpy as np


def one_hot_encode(labels, num_classes):
    """
    :param labels: 레이블 배열
    :param num_classes: 클래스의 수
    :return: one-hot encoding된 레이블 배열
    """
    num_labels = labels.shape[0]  # 레이블의 수
    index_offset = np.arange(num_labels) * num_classes
    one_hot_labels = np.zeros((num_labels, num_classes))
    one_hot_labels.flat[index_offset + labels.ravel()] = 1
    return one_hot_labels


class Softmax:
    def __init__(self):
        self.loss = None  # 손실
        self.y = None     # softmax의 출력
        self.t = None     # 정답 레이블(원-핫 벡터)
        self.crossentropyloss = CrossEntropyLoss()
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis = 1, keepdims = True))
        out = exp / np.sum(exp, axis = 1, keepdims = True)
        return out
    
    def error(self, y_):
        self.y_ = y_
        self.loss = self.crossentropyloss(y_, self.y)
        return self.loss
    
    def forward(self, x):
        self.y = self.softmax(x)
        return self.y
    
    def backward(self, d_y=1):
        batch_size = self.y_.shape[0]
        d_x = (self.y - self.y_)/batch_size
        return d_x
    