MyML
===========
This repository contains simple Machine-learning algorithms built from scratch. The goal is to keep track of my findings and personal skills
 with machine learning in python. In order to allow scalability, each algorithm implements a Classifier, Clustering or DimensionalityReduction object depending on the type of task it performs.

Currently (21/05/2017) the library includes the following methods:
 * simple Perceptron classification


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
If you do not have pip install run the following command:
* easy_install pip

For installing MyML run (you might have to use sudo for super user privileges):
* make install

For running tests:
* make test
