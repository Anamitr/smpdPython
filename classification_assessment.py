import numpy as np

import classifier


def run_crossvalidation(classes, num_of_sets, method_name, method_function, method_extra_args=None):
    pieces, labels = classifier.divide_set_into_num_of_pieces(classes, num_of_sets)
    results = []
    for i in range(0, num_of_sets):
        test_set = pieces[i].T
        test_label_set = labels[i].T

        train_pieces = pieces.copy()
        del train_pieces[i]
        train_set = np.hstack(train_pieces).T

        train_labels = labels.copy()
        del train_labels[i]
        train_labels_set = np.hstack(train_labels).T

        if method_extra_args is not None:
            results.append(method_function(train_set, test_set, train_labels_set, test_label_set, method_extra_args))
        else:
            results.append(method_function(train_set, test_set, train_labels_set, test_label_set))

    print("Average for", method_name, sum(results) / float(len(results)), "%")
    pass
