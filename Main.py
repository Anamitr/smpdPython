from numpy import empty
import numpy as np
import math

import Database
import GlobalVariables

import fisher

classes, classes_names, num_of_traits = Database.getClassesWithNamesAndNumOfTraits()
GlobalVariables.NUM_OF_ALL_FEATURES = num_of_traits

# print(fisher.fisher_one_feature(classes))

# print(fisher.fisher_multiple_features(classes, [0, 2, 3]))

print(fisher.run_fisher_for_num_of_features(classes, 3))
# (7, 15, 33)

print(fisher.run_fisher_with_sfs(classes, 3))
# [30, 15, 7]

# Features: [15, 31, 33]
# Features: [30, 15, 33]