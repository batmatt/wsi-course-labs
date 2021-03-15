import numpy as np


class City:
    def __init__(self, x: int, y: int, name: str):
        # dictonary of distances to other cities
        self.distances = {}

        self.name = name
        self.x = x
        self.y = y

    def fill_distances_dict(self, cities):
        """
        self, cities --> None

        Fills dictonary of distances of the city
        object basing on cities list.
        Method checks whether we took city we are currently
        filling distances dict - if so we put its name
        with value 0. Else we calculate distance to other city.

        """
        for city in cities:
            if city.name == self.name:
                self.distances[city.name] = 0
            else:
                self.distances[city.name] = self.calculate_distance(city)

    def calculate_distance(self, city):
        """
        self, city --> None

        Calculates distance to other city as Euclidean
        distance (which is identical with norm of the vector)

        """
        local_point = np.array((self.x, self.y))
        remote_point = np.array((city.x, city.y))
        return np.linalg.norm(local_point - remote_point)