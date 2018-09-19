import numpy
from scipy.spatial import distance
from collections import Counter

def knn(training_datapoints, training_labels, testing_datapoint, k, type = "classification"):
    print("Hello world from knn")

    assert training_datapoints.shape[0] == training_labels.shape[0], "Different numbers of training points and training labels"
    assert testing_datapoint.shape[0] == training_datapoints.shape[1], "Training and testing point should have the same dimensionality"
    assert k > 0, "k should be greater or equal to 1"
    assert training_datapoints.shape[0] > k-1, "The number of training points should be at least equal to k"
    assert len(testing_datapoint.shape) == 1, "There should be only one testing point"

    print("Hello world from knn after the tests")
    distances = []
    for i in range(0, training_datapoints.shape[0]):
        d = distance.euclidean(training_datapoints[i, :], testing_datapoint)
        distances.append(d)

    sorted_indices = numpy.argsort(numpy.array(distances))

    neighbors_labels = []
    for i in range(0, k):
        neighbors_labels.append(training_labels[sorted_indices[i]])

    print(sorted_indices)
    test_label = 0
    if (type == "regression"):
        print("regression")
        test_label = numpy.mean(neighbors_labels)
    else :
        print("classification")
        count = Counter(neighbors_labels)
        print(count)
        print(count.most_common(1)[0][0])
        test_label = count.most_common(1)[0][0]

    return test_label
