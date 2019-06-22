from numpy import empty
import numpy as np
import math

import Database
import GlobalVariables
import classifier

import fisher
from classification_assessment import run_crossvalidation

TRAIN_SET_RATIO = 0.2
NUM_OF_FEATURES_TO_CHOOSE = 3
NUM_OF_CROSSVALIDATION_ITERATIONS = 10

classes, classes_names, num_of_traits = Database.getClassesWithNamesAndNumOfTraits()
GlobalVariables.NUM_OF_ALL_FEATURES = num_of_traits

# print(fisher.fisher_one_feature(classes))

# print(fisher.fisher_multiple_features(classes, [0, 2, 3]))

# GlobalVariables.CHOSEN_FEATURES = fisher.run_fisher_for_num_of_features(classes, NUM_OF_FEATURES_TO_CHOOSE)
# (7, 15, 33)

GlobalVariables.CHOSEN_FEATURES = fisher.run_fisher_with_sfs(classes, NUM_OF_FEATURES_TO_CHOOSE)
# [30, 15, 7]
print(GlobalVariables.CHOSEN_FEATURES)

X_train, X_test, y_train, y_test = classifier.divide_set(classes, TRAIN_SET_RATIO)

# print("NN:", classifier.classify_with_NN(X_train, X_test, y_train, y_test), "%")
# print("kNN:", classifier.classify_with_kNN(X_train, X_test, y_train, y_test, 3), "%")
print("NM:", classifier.classify_with_NM(X_train, X_test, y_train, y_test), "%")


run_crossvalidation(classes, NUM_OF_CROSSVALIDATION_ITERATIONS, "NM", classifier.classify_with_NM)
