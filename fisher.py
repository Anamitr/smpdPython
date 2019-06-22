import itertools
import sys

import math
import numpy as np

import GlobalVariables


def fisher_one_feature(classes):
    NUM_OF_FEATURES = len(classes[0])
    # print(NUM_OF_FEATURES)
    avg_matrixes = []
    for clazz in classes:
        avg_matrixes.append(clazz.mean(1))
    # print(avg_matrixes)

    variations = []
    for clazz in classes:
        variance = []
        for row in clazz:
            variance.append(math.sqrt(np.var(row)))
            # print(math.sqrt(np.var(row)))
        variations.append(np.array(variance))

    # print("Variations:", variations)

    fishers = []
    for i in range(0, NUM_OF_FEATURES):
        fishers.append(abs(avg_matrixes[0][i] - avg_matrixes[1][i]) / (variations[0][i] + variations[1][i]))
    # print("Fishers:", fishers)
    return max(fishers), fishers.index(max(fishers))


def fisher_multiple_features(classes, features_numbers):
    a_features = classes[0][features_numbers]
    b_features = classes[1][features_numbers]

    a_means = np.array([np.mean(i) for i in a_features])
    b_means = np.array([np.mean(i) for i in b_features])

    a_sample_minus_mean = np.copy(a_features)
    b_sample_minus_mean = np.copy(b_features)
    for i in range(0, len(features_numbers)):
        a_sample_minus_mean[i] = a_sample_minus_mean[i] - a_means[i]
        b_sample_minus_mean[i] = b_sample_minus_mean[i] - b_means[i]

    # np.linalg.det(np.dot(a_sample_minus_mean, a_sample_minus_mean.T) / a_features.shape[1])
    a_covariance_matrix = np.cov(a_features)
    b_covariance_matrix = np.cov(b_features)

    covariance_matrices_det_sum = np.linalg.det(a_covariance_matrix + b_covariance_matrix)
    means_vectors_distance = np.linalg.norm(a_means - b_means)

    result = means_vectors_distance / covariance_matrices_det_sum

    return result


def run_fisher_for_num_of_features(classes, num_of_features):
    combinations = list(itertools.combinations(range(0, GlobalVariables.NUM_OF_ALL_FEATURES), num_of_features))
    print(combinations)

    best_fisher_result = sys.float_info.min
    best_feature_combination = None

    for combination in combinations:
        result = fisher_multiple_features(classes, list(combination))
        print("Combination:", combination, ", result:", result)
        if result > best_fisher_result:
            best_fisher_result = result
            best_feature_combination = combination

    best_feature_combination = list(best_feature_combination)
    best_feature_combination.sort()
    return best_feature_combination


def run_fisher_with_sfs(classes, num_of_features):
    best_score, best_feature = fisher_one_feature(classes)
    best_combination = [best_feature]

    best_fisher_result = sys.float_info.min
    for i in range(1, num_of_features):
        for j in range(0, GlobalVariables.NUM_OF_ALL_FEATURES):
            if j in best_combination:
                continue
            else:
                next_combination = best_combination.copy()
                next_combination.append(j)
                result = fisher_multiple_features(classes, next_combination)
                if result > best_fisher_result:
                    best_fisher_result = result
                    best_feature_num = j
        best_combination.append(best_feature_num)

    best_combination.sort()
    return best_combination
