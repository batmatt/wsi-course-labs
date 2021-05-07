"""
Demonstration of random forest classifier
by
Mateusz Winnicki

Dataset source: https://archive.ics.uci.edu/ml/machine-learning-databases/car/
"""

import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from sklearn.model_selection import (
    train_test_split,
)  # i hope it's not cheating if i use this method just to split dataset neatly
from pandas.core.frame import DataFrame

from confusion_matrix import ConfusionMatrix

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


def prepare_train_and_test_set(
    dataset: DataFrame, train_size: float, is_train_set_sorted: bool = False
):
    train_set, test_set = train_test_split(dataset, train_size=train_size)

    if is_train_set_sorted:
        train_set = train_set.sort_index()

    return train_set, test_set


def get_test_set_classes(test_set: DataFrame):
    classes = {}
    i = 0
    for index in test_set.index:
        if test_set[CLASS][index] not in classes.values():
            classes[i] = test_set[CLASS][index]
            i = i + 1

    return classes


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

                if key == "acc":
                    vgood_occurrences = create_occurrences_lists(
                        feature_values, all_feature_values, occurrences
                    )

        df = pd.DataFrame(
            {
                "unacc_occurrences": unacc_occurrences,
                "acc_occurrences": acc_occurrences,
                "good_occurrences": good_occurrences,
                "vgood_occurrences": vgood_occurrences,
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
        "-hist",
        "--plot-histograms",
        action="store_true",
        help="Plots histograms of loaded data",
    )

    args = parser.parse_args()

    _filename = args.filename
    _plot_histograms = args.plot_histograms

    dataset = load_data_from_file(_filename)

    if _plot_histograms:
        plot_features_histograms(dataset)


if __name__ == "__main__":
    main()
