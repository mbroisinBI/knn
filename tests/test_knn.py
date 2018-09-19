import numpy

import knn


def test_knn_regression():
    print("Test regression")
    training_set = numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    training_labels = numpy.array([1, 2, 3, 4, 5, 6])
    testing_point = numpy.array([0, 0])
    k = 3
    lab = knn.knn(training_set, training_labels, testing_point, k, type="regression")
    assert lab == 2.0

def test_knn_classification():
    print("Test classification")
    training_set = numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]])
    training_labels = numpy.array([1, 3, 3, 4, 5, 6])
    testing_point = numpy.array([0, 0])
    k = 3
    lab = knn.knn(training_set, training_labels, testing_point, k, type="classification")
    assert lab == 3
