import math
import numpy as np
from sklearn.datasets import make_blobs


class CoordinatesGenerator:
    def __init__(self, cities_number: int):
        self.cities_number = cities_number

    def uniform_distribution(self):
        """
        Initializes coordinations of cities in uniform distribution,
        with shape closest to the square.

        """
        M, N = self.find_divisors_nearest_to_square()

        X = np.linspace(0, M * 10, M)
        Y = np.linspace(0, N * 10, N)

        # create grid coordinates based on X and Y linspaces
        mg = np.meshgrid(X, Y)

        # return list of points (x, y)
        return [(x, y) for (x, y) in zip(mg[0].flatten(), mg[1].flatten())]

    def clustered_distribution(self):
        """
        Initializes coordinations of cities grouped into clusters.
        # TODO: Think about some better generator, that is parametrizable but for now i am to exhausted with this lab to do it

        """
        centers = [(10, 10), (20, 20), (30, 30)]
        cluster_std = [2, 2, 2]

        X, y = make_blobs(
            n_samples=30,
            cluster_std=cluster_std,
            centers=centers,
            n_features=2,
            random_state=1,
        )

        # return list of points (x, y)
        return [(x, y) for (x, y) in X]

    def random_distribution(self):
        """
        Initializes coordinations of cities in random distribution.

        """
        X = self.cities_number * np.random.rand(self.cities_number)
        Y = self.cities_number * np.random.rand(self.cities_number)

        # return list of points (x, y)
        return [(x, y) for (x, y) in zip(X, Y)]

    def find_divisors_nearest_to_square(self):
        """
        Finds pair of divisors of number of cities that is closest
        to the square. Thanks to that uniform_distriubtion() does not
        generate elongated rectangles.

        """
        M = math.ceil(math.sqrt(self.cities_number))
        N = int(self.cities_number / M)

        while N * M != float(self.cities_number):
            M -= 1
            N = int(self.cities_number / M)

        return M, N
