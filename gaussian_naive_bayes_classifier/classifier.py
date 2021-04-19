import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)  # i hope it's not cheating if i use this method just to split dataset neatly
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


def calculate_statistics_by_class(train_set: DataFrame):
    groups = train_set.groupby(CLASS)

    mean_values_by_class = {}
    std_deviation_values_by_class = {}
    prior_probability_by_class = {}

    for class_name, group in groups:
        mean_values_by_class[class_name] = group[[X1, X2, X3, X4]].mean()
        std_deviation_values_by_class[class_name] = group[[X1, X2, X3, X4]].std()
        prior_probability_by_class[class_name] = len(group) / len(train_set)

    return (
        mean_values_by_class,
        std_deviation_values_by_class,
        prior_probability_by_class,
    )


def gaussian_density_function(x: float, mean: float, std_deviation: float) -> float:
    exponent = math.exp(-((x - mean) ** 2 / (2 * std_deviation ** 2)))

    return exponent / (std_deviation * math.sqrt(2 * math.pi))


def calculate_posterior_probabilities(
    feature_vector: list,
    mean_values: list,
    std_deviation_values: list,
    prior_probabilities: list,
) -> dict:
    posterior_probabilities = {}
    features_number = len(feature_vector)

    for class_name in mean_values.keys():
        likelihood = 1

        for index in range(features_number):
            feature = feature_vector[index]
            mean = mean_values[class_name][index]
            std_deviation = std_deviation_values[class_name][index]

            gaussian_probability = gaussian_density_function(
                feature, mean, std_deviation
            )
            likelihood = likelihood * gaussian_probability

        prior_probability = prior_probabilities[class_name]
        joint_probability = prior_probability * likelihood
        posterior_probabilities[class_name] = joint_probability

    return posterior_probabilities


def make_prediction(
    feature_vector: list,
    mean_values: list,
    std_deviation_values: list,
    prior_probabilities: list,
) -> str:
    posterior_probabilities = calculate_posterior_probabilities(
        feature_vector, mean_values, std_deviation_values, prior_probabilities
    )
    prediction = max(posterior_probabilities, key=posterior_probabilities.get)

    return prediction


def test_set_classification(
    test_set: DataFrame,
    mean_values: list,
    std_deviation_values: list,
    prior_probabilities: list,
):
    results = pd.DataFrame(columns=["prediction", "label", "result"])

    for index, row in test_set.iterrows():
        feature_vector = row.to_list()

        # extract label
        label = feature_vector[-1]
        # remove label from feature vector
        feature_vector.pop()

        prediction = make_prediction(
            feature_vector,
            mean_values,
            std_deviation_values,
            prior_probabilities,
        )

        if prediction == label:
            results = results.append(
                {"prediction": prediction, "label": label, "result": 1},
                ignore_index=True,
            )
        else:
            results = results.append(
                {"prediction": prediction, "label": label, "result": 0},
                ignore_index=True,
            )

    return results


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


def prepare_train_and_test_set(
    dataset: DataFrame, train_size: float, is_train_set_sorted: bool = False
):
    train_set, test_set = train_test_split(dataset, train_size=train_size)

    if is_train_set_sorted:
        train_set = train_set.sort_index()

    return train_set, test_set


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
        "-srt",
        "--sort-train-set",
        action="store_true",
        help="Sorts train set after spliting dataset to train and test set",
    )

    args = parser.parse_args()

    _filename = args.filename
    _train_size = args.train_size
    _sort_train_set = args.sort_train_set

    dataset = load_data_from_file(_filename)

    # uncomment to check grouping classes by two chosen features
    # plot_labels_by_two_features(X1, X2, dataset)

    train_set, test_set = prepare_train_and_test_set(
        dataset, _train_size, _sort_train_set
    )

    (
        mean_values,
        std_variation_values,
        prior_probabilities,
    ) = calculate_statistics_by_class(train_set)

    results = test_set_classification(
        test_set, mean_values, std_variation_values, prior_probabilities
    )

    print(results)


if __name__ == "__main__":
    main()
