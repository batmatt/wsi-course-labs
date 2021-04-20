import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame

CLASSES = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


class ConfusionMatrix:
    def __init__(self, results: DataFrame):
        self.confusion_matrix = pd.crosstab(
            results["actual value"],
            results["prediction"],
            rownames=["Actual value"],
            colnames=["Predicted"],
        )

        self.tp = {}
        self.tn = {}
        self.fp = {}
        self.fn = {}

        for i in range(len(CLASSES)):
            self.tp[CLASSES[i]] = self.confusion_matrix.loc[CLASSES[i]][CLASSES[i]]

            self.tn[CLASSES[i]] = sum(
                [
                    self.confusion_matrix.loc[CLASSES[x]][CLASSES[y]]
                    for x, y in np.ndindex(len(CLASSES), len(CLASSES))
                    if x != i and y != i
                ]
            )

            self.fp[CLASSES[i]] = sum(
                [
                    self.confusion_matrix.loc[CLASSES[x]][CLASSES[i]]
                    for x in range(len(CLASSES))
                    if x != i
                ]
            )

            self.fn[CLASSES[i]] = sum(
                [
                    self.confusion_matrix.loc[CLASSES[i]][CLASSES[x]]
                    for x in range(len(CLASSES))
                    if x != i
                ]
            )

        self.overall_accuracy = (sum(self.tp.values()) + sum(self.tn.values())) / (
            sum(self.tp.values())
            + sum(self.tn.values())
            + sum(self.fp.values())
            + sum(self.fn.values())
        )

        self.metrics = {}

    def plot_confusion_matrix(self):
        sn.heatmap(self.confusion_matrix, annot=True, cmap="Blues", fmt="g")
        plt.show()

    def calculate_metrics_by_classes(self):
        for i in range(len(CLASSES)):
            self.metrics[CLASSES[i]] = {
                "precision": self.tp[CLASSES[i]]
                / (self.tp[CLASSES[i]] + self.fp[CLASSES[i]]),
                "recall": self.tp[CLASSES[i]]
                / (self.tp[CLASSES[i]] + self.fn[CLASSES[i]]),
                "fall-out": self.fp[CLASSES[i]]
                / (self.fp[CLASSES[i]] + self.tn[CLASSES[i]]),
            }
