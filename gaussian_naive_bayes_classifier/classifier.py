import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame

X1 = "sepal length [cm]"
X2 = "sepal width [cm]"
X3 = "petal length [cm]"
X4 = "petal width [cm]"
CLASS = "class"


def load_data_from_file(filename: str) -> DataFrame:
    df = pd.read_csv(
        filename,
        sep=",",
        names=[X1, X2, X3, X4, CLASS],
    )

    return df


def calculate_statistics(dataset: DataFrame):
    dataset_mean = dataset[[X1, X2, X3, X4]].mean()

    dataset_std_deviation = dataset[[X1, X2, X3, X4]].std()

    return dataset_mean, dataset_std_deviation


def plot_labels_by_two_features(
    feature1: str, feature2: str, dataset: DataFrame
) -> None:
    cut_dataset = dataset[[feature1, feature2, CLASS]]

    groups = cut_dataset.groupby(CLASS)

    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(
            group[feature1], group[feature2], marker="o", linestyle="", ms=5, label=name
        )
    plt.title(f"Classification by {feature1} and {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    ax.legend()

    plt.savefig(f"plots/{feature1}_{feature2}.png")
    plt.clf()


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
        "--training-size",
        type=float,
        help="Percentage of dataset used in the training set.",
    )

    args = parser.parse_args()

    _filename = args.filename
    _training_size = args.training_size

    dataset = load_data_from_file(_filename)
    statistics = calculate_statistics(dataset)

    # uncomment to check grouping by two chosen features
    plot_labels_by_two_features(X1, X2, dataset)


if __name__ == "__main__":
    main()
