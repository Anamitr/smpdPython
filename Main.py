from numpy import empty
import numpy as np
import math

import Database
import GlobalVariables
import classifier

import fisher

TRAIN_SET_RATIO = 0.2
NUM_OF_FEATURES_TO_CHOOSE = 3

classes, classes_names, num_of_traits = Database.getClassesWithNamesAndNumOfTraits()
GlobalVariables.NUM_OF_ALL_FEATURES = num_of_traits

# print(fisher.fisher_one_feature(classes))

# print(fisher.fisher_multiple_features(classes, [0, 2, 3]))

# GlobalVariables.CHOSEN_FEATURES = fisher.run_fisher_for_num_of_features(classes, NUM_OF_FEATURES_TO_CHOOSE)
# (7, 15, 33)

GlobalVariables.CHOSEN_FEATURES = fisher.run_fisher_with_sfs(classes, NUM_OF_FEATURES_TO_CHOOSE)
# [30, 15, 7]
print(GlobalVariables.CHOSEN_FEATURES)

# print("NN:", classifier.classify_with_NN(classes, TRAIN_SET_RATIO), "%")
print("kNN:", classifier.classify_with_kNN(classes, TRAIN_SET_RATIO, 3), "%")
# print("NM:", classifier.classify_with_NM(classes, TRAIN_SET_RATIO), "%")
# 85.35031847133759




# all_data = classifier.label_data(classes)
# classifier.divide_set(all_data, 0.3)

# Features: [15, 31, 33]
# Features: [30, 15, 33]
