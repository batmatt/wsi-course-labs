from typing import Dict
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.frame import DataFrame


class ConfusionMatrix:
    def __init__(self, results: DataFrame, classes: Dict):
        self.classes = classes

        self.confusion_matrix = pd.crosstab(
            results["actual value"],
            results["prediction"],
            rownames=["Actual value"],
            colnames=["Predicted"],
        )

        # fill confusion matrix with empty predicted columns if there are missing
        for value in results["actual value"].unique():
            if value not in results["prediction"].unique():
                predicted = [0, 0, 0, 0]
                self.confusion_matrix[value] = predicted

        # row names are sorted, so to make crosstab work properly, we need to rearrange columns too
        sorted_columns = self.confusion_matrix.columns.sort_values()
        self.confusion_matrix = self.confusion_matrix.reindex(
            sorted_columns, axis="columns"
        )

        self.tp = {}
        self.tn = {}
        self.fp = {}
        self.fn = {}

        for i in range(len(classes)):
            self.tp[classes[i]] = self.confusion_matrix.loc[classes[i]][classes[i]]

            self.tn[classes[i]] = sum(
                [
                    self.confusion_matrix.loc[classes[x]][classes[y]]
                    for x, y in np.ndindex(len(classes), len(classes))
                    if x != i and y != i
                ]
            )

            self.fp[classes[i]] = sum(
                [
                    self.confusion_matrix.loc[classes[x]][classes[i]]
                    for x in range(len(classes))
                    if x != i
                ]
            )

            self.fn[classes[i]] = sum(
                [
                    self.confusion_matrix.loc[classes[i]][classes[x]]
                    for x in range(len(classes))
                    if x != i
                ]
            )

        self.accuracy = (sum(self.tp.values()) + sum(self.tn.values())) / (
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
        for i in range(len(self.classes)):
            precision_denominator = self.tp[self.classes[i]] + self.fp[self.classes[i]]

            if precision_denominator != 0:
                self.metrics[self.classes[i]] = {
                    "precision": self.tp[self.classes[i]]
                    / (self.tp[self.classes[i]] + self.fp[self.classes[i]]),
                    "recall": self.tp[self.classes[i]]
                    / (self.tp[self.classes[i]] + self.fn[self.classes[i]]),
                    "fall-out": self.fp[self.classes[i]]
                    / (self.fp[self.classes[i]] + self.tn[self.classes[i]]),
                }
            else:
                self.metrics[self.classes[i]] = {
                    "precision": 0.0,
                    "recall": self.tp[self.classes[i]]
                    / (self.tp[self.classes[i]] + self.fn[self.classes[i]]),
                    "fall-out": self.fp[self.classes[i]]
                    / (self.fp[self.classes[i]] + self.tn[self.classes[i]]),
                }
