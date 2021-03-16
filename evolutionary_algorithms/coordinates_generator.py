import math
import numpy as np


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
        mg = np.meshgrid(X, Y)

        return [(x, y) for (x, y) in zip(mg[0].flatten(), mg[1].flatten())]

    def clustered_distribution(self):
        """
        Initializes coordinations of cities grouped into clusters.

        """

    def random_distribution(self):
        """
        Initializes coordinations of cities in random distribution.

        """
        X = self.cities_number * np.random.rand(self.cities_number)
        Y = self.cities_number * np.random.rand(self.cities_number)

        print(X)

        return zip(X, Y)

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
