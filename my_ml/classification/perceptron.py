import numpy as np
from myML.machine_learning import Classifier


class Perceptron(Classifier):
    '''
    This class implements a simple Perceptron classifier
    '''
    def __init__(self):
        self.weights = []

    def predict(self, x):
        '''
        Predicts an output binary class (0 or 1) for a given input attribute
        vector 'x'.
        :param x: list of attribute values
        :type x: list
        :return: 0 or 1
        '''
        activation = self.weights[0] + np.sum(self.weights[1:] * np.array(x))
        return 1.0 if activation >= 0.0 else 0.0

    def fit(self, x_train, y_train, l_rate, n_iter):
        '''
        Adapts the weigths of the perceptron model to fit the input data
        'x_train' as defined by 'y_train'.
        :param x_train:dataset used for training
        :type x_train: list_list
        :param y_train: expected output classes (0 or 1)
        :type y_train: list
        :param l_rate: degree in which  weights are corrected in each training
        iteration.
        :param n_iter: number of times to iterate through the training examples
        while updating weights
        :return: the weights that fit the input data
        :rtype: list
        '''
        assert len(x_train) == len(y_train), \
            'training data and labels size mismatch: {0} != {1}'.format(
                len(x_train), len(y_train)
            )

        weights = [0.0 for i in range(len(x_train[0])+1)]
        for epoch in range(n_iter):
            for row, expec in zip(x_train, y_train):
                self.weights = weights
                prediction = self.predict(row)
                error = expec - prediction
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row)):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        return weights
