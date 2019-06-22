import numpy as np


# Obróć to !!!!!
def getClassesWithNamesAndNumOfTraits():
    classes = []
    classes_names = []
    num_of_traits = int

    with open("Maple_Oak.txt", "r") as file:
        data = file.readlines()
        num_of_traits = int(data[0].split(", ")[0])
        # print(num_of_traits)

        del data[0]
        # print(data)
        for row in data:
            row = row.replace("\n", "")
            row = row.split(",")
            # print(row[0])
            class_name = row[0].split(" ")[0]
            # print(class_name)
            del row[0]
            row = [float(i) for i in row]

            if class_name not in classes_names:
                classes_names.append(class_name)
                classes.append([])
            classes[classes_names.index(class_name)].append(row)
    # print(classes)
    classes[0] = np.transpose(np.array(classes[0]))
    classes[1] = np.transpose(np.array(classes[1]))
    return [classes, classes_names, num_of_traits]


def getExampleData():
    classes = [
        [
            [1, 2, 3, 1],
            [2, 2, 3, 3],
            [-2, -1, -2, -2]
        ],
        [
            [2, 3, 3, 3],
            [4, 3, 1, 5],
            [1, 0, 1, 1]
        ]
    ]
    classes = np.array(classes)
    return [classes, ["a", "b"], 3]


def getExampleData2():
    classes = [
        [
            [-2, 0, -1, -2, 0],
            [2, 5, 3, 2, 3],
            [1, 1, 2, 1, 0],
            [3, 4, 4, 4, 5]
        ],
        [
            [6, 5, 7, 5, 2],
            [1, 2, 1, 1, 0],
            [-1, 0, 1, 1, -1],
            [-2, -1, 0, -1, -1]
        ]
    ]
    classes = np.array(classes)
    return [classes, ["a", "b"], 4]
