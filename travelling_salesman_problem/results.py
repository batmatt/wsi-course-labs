import numpy as np


class Results:
    def __init__(self):
        # lists of results
        self.best_lengths = []
        self.best_length_generations = []
        # statictical metrics
        self.minimum_length = 0
        self.maximum_length = 0
        self.mean_length = 0
        self.std_deviation_length = 0
        self.minimum_generation = 0
        self.maximum_generation = 0
        self.mean_generation = 0
        self.std_deviation_generation = 0

    def __str__(self) -> str:
        metrics_repr = (
            f"\nMetrics after {len(self.best_lengths)} tries\nMinimum length: {self.minimum_length}\nMaximum length: {self.maximum_length}\n"
            + f"Avarage length: {self.mean_length}\nStandard deviation of lengths: {self.std_deviation_length}\n"
            + f"Minimum generation: {self.minimum_generation}\nMaximum generation: {self.maximum_generation}\n"
            + f"Avarage generation: {self.mean_generation}\nStandard deviation of generations: {self.std_deviation_generation}"
        )
        return metrics_repr

    def update_results_lists(self, best_length: float, generation: int):
        self.best_lengths.append(best_length)
        self.best_length_generations.append(generation)

    def calculate_metrics(self):
        sum = 0

        for x in self.best_lengths:
            sum = sum + x

        self.minimum_length = min(self.best_lengths)
        self.maximum_length = max(self.best_lengths)
        self.mean_length = sum / len(self.best_lengths)
        self.std_deviation_length = np.std(self.best_lengths)

        sum = 0

        for x in self.best_length_generations:
            sum = sum + x

        self.minimum_generation = min(self.best_length_generations)
        self.maximum_generation = max(self.best_length_generations)
        self.mean_generation = sum / len(self.best_length_generations)
        self.std_deviation_generation = np.std(self.best_length_generations)
