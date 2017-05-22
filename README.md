MyML
===========
MyML will implement simple machine-learning algorithms for classification and clustering, as well as model evaluation tools such as performance measures, cross-validation and tests. The goal of this repo is to keep track of my findings and personal skills with machine learning in python.

Algorithms are 



This repository contains simple Machine-learning algorithms built from scratch as well as tools for evaluating models (performance measures, cross-validation, tests). The goal is to keep track of my findings and personal skills
 with machine learning in python. In order to allow scalability, each algorithm implements a Classifier, Clustering or DimensionalityReduction object depending on the type of task it performs.

Currently (21/05/2017) the library includes the following methods:
 * Perceptron or Single-layer perceptron: is an algorithm for supervised learning of binary classifiers. This kind of classifiers decide whether an input, represented by a vector of numbers, belongs to some specific class (1) or not (0). It is a type of linear classifier, which means that the classification algorithm that makes its predictions is based on a linear predictor function (activation function) combining a set of weights with the feature vector, just like in linear regression. The algorithm allows for online learning, in that it processes elements in the training set one at a time. 
 The Perceptron is inspired by the information processing of a single neural cell called a neuron. A neuron accepts input signals via its dendrites, which pass the electrical signal down to the cell body. In a similar way, the Perceptron receives input signals from examples of training data that we weight and combine in the activation function. The activation is then transformed into an output value or prediction using a transfer function, such as the step transfer function or sigmoid function. In this way, the Perceptron is a classification algorithm for problems with two classes (0 and 1) where a linear equation (hyperplane) can be used to separate the two classes. It is closely related to linear regression and logistic regression that make predictions in a similar way (e.g. a weighted sum of inputs). The weights of the Perceptron algorithm must be estimated from your training data using stochastic gradient descent.
 
 

Usage example
=========
    from myML.classification.perceptron import Perceptron

    x_train =  [
                [2.7810836, 2.550537003],
                [1.465489372, 2.362125076],
                [3.396561688, 4.400293529],
                [1.38807019, 1.850220317],
                [3.06407232, 3.005305973],
                [7.627531214, 2.759262235],
                [5.332441248, 2.088626775],
                [6.922596716, 1.77106367],
                [8.675418651, -0.242068655],
                [7.673756466,   3.508563011]
            ]
    y_train =  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    l_rate = 0.1
    iter = 5
    perceptron = Perceptron()
    perceptron.fit(x_train, y_train, l_rate, iter)
    perceptron.predict([2.7810836,2.550537003])


Install
=========
If you do not have pip installed in your machine, run the following command:
* easy_install pip

For installing MyML run (you might have to use sudo for super user privileges):
* make install

For running tests:
* make test
