"""
Demonstration of random forest classifier
by
Mateusz Winnicki

Dataset source: https://archive.ics.uci.edu/ml/machine-learning-databases/car/
"""

import argparse
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import (
    train_test_split,
)  # i hope it's not cheating if i use this method just to split dataset neatly
from pandas.core.frame import DataFrame

from random_forest_confusion_matrix import ConfusionMatrix

X1 = "buying"
X2 = "maint"
X3 = "doors"
X4 = "persons"
X5 = "lug_boot"
X6 = "safety"
CLASS = "class"

FEATURES = [X1, X2, X3, X4, X5, X6]


def load_data_from_file(filename: str) -> DataFrame:
    df = pd.read_csv(
        filename,
        sep=",",
        names=[X1, X2, X3, X4, X5, X6, CLASS],
    )

    return df


def get_dataset_classes(dataset: DataFrame):
    classes = {}
    i = 0
    for index in dataset.index:
        if dataset[CLASS][index] not in classes.values():
            classes[i] = dataset[CLASS][index]
            i = i + 1

    return classes


def evaluate_with_cross_validation(
    train_set: DataFrame, folds_number: int, trees_number: int, max_depth: int
):
    split = np.array_split(train_set, folds_number)
    confusion_matrices = []

    for i in range(folds_number):
        train_set_parts = split.copy()
        del train_set_parts[i]

        train_set = pd.concat(train_set_parts, sort=False)
        test_set = split[i]

        classes = get_dataset_classes(train_set)
        results = random_forest_classification(
            train_set, test_set, trees_number, max_depth
        )

        cm = ConfusionMatrix(results, classes)
        cm.calculate_metrics_by_classes()
        confusion_matrices.append(cm)

    accuracy_values = []
    precision_values = {}
    recall_values = {}
    for matrix in confusion_matrices:
        accuracy_values.append(matrix.accuracy)
        for class_name in matrix.metrics:
            if class_name not in precision_values:
                precision_values[class_name] = []
                recall_values[class_name] = []
            precision_values[class_name].append(matrix.metrics[class_name]["precision"])
            recall_values[class_name].append(matrix.metrics[class_name]["recall"])

    print(
        f"Classifier mean overall accuracy after cross_validation:\nMin: {min(accuracy_values)}\nMax: {max(accuracy_values)}\n"
        + f"Mean: {np.mean(accuracy_values)}\nStd deviation: {np.std(accuracy_values)}\n"
    )
    for class_name in precision_values:
        print(
            f"Mean {class_name} precision statistics after cross_validation:\nMin: {min(precision_values[class_name])}\nMax: {max(precision_values[class_name])}\n"
            + f"Mean: {np.mean(precision_values[class_name])}\nStd deviation: {np.std(precision_values[class_name])}\n"
        )
        print(
            f"Mean {class_name} recall statistics after cross_validation:\nMin: {min(recall_values[class_name])}\nMax: {max(recall_values[class_name])}\n"
            + f"Mean: {np.mean(recall_values[class_name])}\nStd deviation: {np.std(recall_values[class_name])}\n"
        )


def random_forest_classification(
    train_set: DataFrame, test_set: DataFrame, trees_number: int, max_depth: int
):
    trees = []
    for i in range(trees_number):
        tree = build_decision_tree(train_set, max_depth)
        trees.append(tree)

    results = pd.DataFrame()
    for _, row in test_set.iterrows():
        prediction = make_forest_prediction(row, trees)

        results = results.append(
            {"prediction": prediction, "actual value": row[CLASS]},
            ignore_index=True,
        )

    return results


def test_split(dataset: DataFrame, feature: str, value: str):
    left = dataset[dataset[feature] == value]
    right = dataset[dataset[feature] != value]

    return left, right


def calculate_gini_index(left: DataFrame, right: DataFrame, class_values: list):
    gini_index = 0.0
    branches = [left, right]
    dataset_size = len(left.index) + len(right.index)
    classes = list(set(class_values))

    for branch in branches:
        branch_size = len(branch.index)
        if branch_size == 0:
            continue
        score = 0.0
        for class_value in classes:
            class_value_occurrences = branch.loc[
                branch[CLASS] == class_value, CLASS
            ].count()
            proportion = class_value_occurrences / branch_size
            score += proportion * proportion

        gini_index += (1.0 - score) * (branch_size / dataset_size)

    return gini_index


def get_best_split(dataset: DataFrame):
    class_values = dataset[CLASS].tolist()
    previous_best_gini_index = 10000
    best_feature = ""
    best_value = ""
    best_left = ""
    best_right = ""

    random_features = random.sample(FEATURES, 2)

    for feature in random_features:
        values = list(dict.fromkeys(dataset[feature].tolist()))

        for value in values:
            left, right = test_split(dataset, feature, value)
            gini_index = calculate_gini_index(left, right, class_values)

            if gini_index < previous_best_gini_index:
                previous_best_gini_index = gini_index
                best_feature = feature
                best_value = value
                best_left = left
                best_right = right

    best_split = {
        "feature": best_feature,
        "value": best_value,
        "left": best_left,
        "right": best_right,
    }
    return best_split


def terminal_node(branch: DataFrame):
    labels = [row[CLASS] for _, row in branch.iterrows()]
    return max(set(labels), key=labels.count)


def split_on_node(node: dict, max_depth: int, current_depth: int = 1):
    left, right = node["left"], node["right"]
    del node["left"]
    del node["right"]

    # if tree has only right branches go stright to terminal node on the right
    if len(left.index) == 0:
        node["left"] = node["right"] = terminal_node(right)
        return

    # if tree has only left branches go stright to terminal node on the left
    if len(right.index) == 0:
        node["left"] = node["right"] = terminal_node(left)
        return

    # if depth is equal maximum depth given by user, make nodes terminal
    if current_depth >= max_depth:
        node["left"], node["right"] = terminal_node(left), terminal_node(right)
        return

    # tree grows non terminal node on the left
    node["left"] = get_best_split(left)
    split_on_node(node["left"], max_depth, current_depth + 1)

    # tree grows non terminal node on the right
    node["right"] = get_best_split(right)
    split_on_node(node["right"], max_depth, current_depth + 1)


def build_decision_tree(train_set: DataFrame, max_depth: int):
    tree_root = get_best_split(train_set)
    split_on_node(tree_root, max_depth)

    return tree_root


def print_tree(node: dict, depth: int = 0, side: str = "  "):
    # if current node is a dictionary it means it isn't terminal node yet
    if isinstance(node, dict):
        print(f"{depth * 2 * ' '}{side}{node['feature']} == {node['value']}")
        print_tree(node["left"], depth + 1, "L ")
        print_tree(node["right"], depth + 1, "R ")
    else:
        print(f"{depth * 2 * ' '}{side}[{node}]")


def make_prediction(feature_vector: list, node: dict):
    if str(feature_vector[node["feature"]]) != node["value"]:
        if isinstance(node["left"], dict):
            return make_prediction(feature_vector, node["left"])
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return make_prediction(feature_vector, node["right"])
        else:
            return node["right"]


def make_forest_prediction(feature_vector: list, forest: list):
    predictions = []
    for tree in forest:
        predictions.append(make_prediction(feature_vector, tree))

    return max(set(predictions), key=predictions.count)


def plot_features_histograms(dataset: DataFrame):
    groups = dataset.groupby(CLASS)

    feature_values_occurrences_by_class = {}
    for name, group in groups:
        for feature in FEATURES:
            feature_values = group[feature].value_counts().index.tolist()
            feature_values_occurrences = []

            for i in range(len(feature_values)):
                feature_values_occurrences.append(
                    {
                        feature_values[i]: group[feature].value_counts()[
                            feature_values[i]
                        ]
                    }
                )

            try:
                feature_values_occurrences_by_class[feature].append(
                    {name: feature_values_occurrences}
                )
            except KeyError:
                feature_values_occurrences_by_class[feature] = []
                feature_values_occurrences_by_class[feature].append(
                    {name: feature_values_occurrences}
                )

    for feature in feature_values_occurrences_by_class.keys():
        fig, ax = plt.subplots()

        all_feature_values = []
        unacc_occurrences = []
        acc_occurrences = []
        good_occurrences = []
        vgood_occurrences = []

        for class_with_values_occurrences in feature_values_occurrences_by_class[
            feature
        ]:

            for key, value in class_with_values_occurrences.items():

                for feature_value_with_occurrences in value:
                    feature_value_occurrence = list(
                        feature_value_with_occurrences.items()
                    )[0]
                    feature_values.append(feature_value_occurrence[0])

                    if feature_value_occurrence[0] not in all_feature_values:
                        all_feature_values.append(feature_value_occurrence[0])

        for class_with_values_occurrences in feature_values_occurrences_by_class[
            feature
        ]:
            occurrences = []
            feature_values = []

            for key, value in class_with_values_occurrences.items():

                for feature_value_with_occurrences in value:
                    feature_value_occurrence = list(
                        feature_value_with_occurrences.items()
                    )[0]
                    feature_values.append(feature_value_occurrence[0])
                    occurrences.append(feature_value_occurrence[1])

                if key == "unacc":
                    unacc_occurrences = create_occurrences_lists(
                        feature_values, all_feature_values, occurrences
                    )
                if key == "acc":
                    acc_occurrences = create_occurrences_lists(
                        feature_values, all_feature_values, occurrences
                    )
                if key == "good":
                    good_occurrences = create_occurrences_lists(
                        feature_values, all_feature_values, occurrences
                    )

                if key == "vgood":
                    vgood_occurrences = create_occurrences_lists(
                        feature_values, all_feature_values, occurrences
                    )

        df = pd.DataFrame(
            {
                "unacc": unacc_occurrences,
                "acc": acc_occurrences,
                "good": good_occurrences,
                "vgood": vgood_occurrences,
            },
            index=all_feature_values,
        )
        ax = df.plot.bar(rot=0)

        plt.xlabel("Feature value")
        plt.ylabel("Number of value occurences")
        plt.title(f"Histogram of feature: {feature}")
        plt.savefig(f"plots/{feature}_histogram.png")
        plt.clf()


def create_occurrences_lists(
    feature_values: list, all_feature_values: list, occurrences: list
):
    filled_occurrences = []
    for value in all_feature_values:
        if value not in feature_values:
            filled_occurrences.append(0)
        else:
            filled_occurrences.append(occurrences[feature_values.index(value)])

    return filled_occurrences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        help="Path to file with data",
    )

    parser.add_argument(
        "-ts",
        "--train-size",
        type=float,
        help="Percentage of dataset used in the train set.",
    )

    parser.add_argument(
        "-fn",
        "--folds_number",
        type=int,
        help="Number of folds used in cross-validation",
    )

    parser.add_argument(
        "-md",
        "--max-depth",
        type=int,
        help="Maximum depth of decision trees",
    )

    parser.add_argument(
        "-tn",
        "--trees-number",
        type=int,
        help="Number of trees used in random forest",
    )

    parser.add_argument(
        "-hist",
        "--plot-histograms",
        action="store_true",
        help="Plots histograms of loaded data",
    )

    args = parser.parse_args()

    _filename = args.filename
    _train_size = args.train_size
    _folds_number = args.folds_number
    _max_depth = args.max_depth
    _trees_number = args.trees_number
    _plot_histograms = args.plot_histograms

    dataset = load_data_from_file(_filename)

    if _plot_histograms:
        plot_features_histograms(dataset)

    train_set, validation_set = train_test_split(dataset, train_size=_train_size)
    classes = get_dataset_classes(train_set)

    # first evaluation with cross validation
    evaluate_with_cross_validation(train_set, _folds_number, _trees_number, _max_depth)

    # second evaluation with train and validation set
    results = random_forest_classification(
        train_set, validation_set, _trees_number, _max_depth
    )

    cm = ConfusionMatrix(results, classes)
    cm.calculate_metrics_by_classes()
    cm.plot_confusion_matrix()

    accuracy = cm.accuracy
    precision_values = {}
    recall_values = {}

    for class_name in cm.metrics:
        if class_name not in precision_values:
            precision_values[class_name] = []
            recall_values[class_name] = []
        precision_values[class_name].append(cm.metrics[class_name]["precision"])
        recall_values[class_name].append(cm.metrics[class_name]["recall"])

    print(f"Classifier accuracy after evaluation on validation set: {accuracy}")
    for class_name in precision_values:
        print(
            f"{class_name} precision after evaluation on validation set: {precision_values[class_name]}"
        )
        print(
            f"{class_name} recall after evaluation on validation set: {recall_values[class_name]}"
        )


if __name__ == "__main__":
    main()
