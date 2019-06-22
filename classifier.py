import heapq

import numpy as np
import sklearn as sklearn
from sklearn.model_selection import train_test_split

import GlobalVariables


def divide_set(classes, training_part_procentage):
    all_data = trim_and_label_data(classes)
    X_train, X_test, y_train, y_test = train_test_split(all_data[1:].T, all_data[:1].T,
                                                        test_size=training_part_procentage, random_state=42)
    return X_train, X_test, y_train, y_test
    # return X_train.T, X_test.T, y_train.T, y_test.T


def trim_and_label_data(classes):
    classA = np.insert(classes[0], 0, np.array([0 for i in classes[0][0]]), 0)
    classB = np.insert(classes[1], 0, np.array([1 for i in classes[1][0]]), 0)
    all_data = np.append(classA, classB, 1)
    chosen_rows = [x + 1 for x in GlobalVariables.CHOSEN_FEATURES]
    chosen_rows.insert(0, 0)
    return all_data[chosen_rows]  # all_data[[x+1 for x in GlobalVariables.CHOSEN_FEATURES].append(0)]


def classify_with_NN(classees, training_part_procentage):
    X_train, X_test, y_train, y_test = divide_set(classees, training_part_procentage)
    correct_matches = 0

    for i in range(0, len(X_test)):
        neirest_neighbour_num = 0
        best_distance = np.linalg.norm(X_test[i] - X_train[0])
        for j in range(1, len(X_train)):
            distance = np.linalg.norm(X_test[i] - X_train[j])
            if distance < best_distance:
                best_distance = distance
                neirest_neighbour_num = j
        if y_test[i] == y_train[neirest_neighbour_num]:
            correct_matches += 1

    return correct_matches / len(X_test) * 100


def classify_with_kNN(classees, training_part_procentage, k):
    X_train, X_test, y_train, y_test = divide_set(classees, training_part_procentage)
    correct_matches = 0

    for i in range(0, len(X_test)):
        results = []
        for j in range(0, len(X_train)):
            results.append([np.linalg.norm(X_test[i] - X_train[j]), j, y_train[j]])
        results = np.array(results)
        results = results[results[:, 0].argsort()]

        correct_neighbours = len([1 for i in range(0, k) if results[i][2] == y_test[i]])
        correct_matches += 1 if correct_neighbours > k / 2 else 0
        pass

    return correct_matches / len(X_test) * 100


def classify_with_NM(classees, training_part_procentage):
    X_train, X_test, y_train, y_test = divide_set(classees, training_part_procentage)
    correct_matches = 0

    classA = np.array([X_train[i] for i in range(0, len(X_train)) if y_train[i] == 0])
    classB = np.array([X_train[i] for i in range(0, len(X_train)) if y_train[i] == 1])

    meanA = classA.mean(0)
    meanB = classB.mean(0)

    for i in range(0, len(X_test)):
        distanceA = np.linalg.norm(X_test[i] - meanA)
        distanceB = np.linalg.norm(X_test[i] - meanB)
        if distanceA < distanceB and y_test[i] == 0 or distanceA > distanceB and y_test[i] == 1:
            correct_matches += 1

    return correct_matches / len(X_test) * 100


