from unittest import TestCase
from my_ml.classification.perceptron import Perceptron


class TestPerceptron(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.perceptron = Perceptron()
        cls.x_train = [
            [2.7810836, 2.550537003],
            [1.465489372, 2.362125076],
            [3.396561688, 4.400293529],
            [1.38807019, 1.850220317],
            [3.06407232, 3.005305973],
            [7.627531214, 2.759262235],
            [5.332441248, 2.088626775],
            [6.922596716, 1.77106367],
            [8.675418651, -0.242068655],
            [7.673756466, 3.508563011]
        ]
        cls.y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        cls.weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
        cls.l_rate = 0.1
        cls.iter = 5

    def test_predict(self):
        for x_item, y_item in zip(self.x_train, self.y_train):
            self.perceptron.weights = self.weights
            predicted = self.perceptron.predict(x_item)
            self.assertEquals(y_item, predicted)

    def test_fit(self):
        got = self.perceptron.fit(
            self.x_train, self.y_train, self.l_rate, self.iter
        )
        self.assertEquals(self.weights, got)

    def test_dump(self):
        import os
        self.perceptron.dump(name='dump_test.pkl')
        self.assertTrue(os.path.exists('models/dump_test.pkl'))

    def test_load(self):
        perceptron = self.perceptron.load(infile='models/dump_test.pkl')
        self.assertTrue(isinstance(perceptron, self.perceptron.__class__))
