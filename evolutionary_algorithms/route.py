import random


class Route:
    def __init__(self, cities):
        # shuffle list of cities in random order
        self.sequence_of_cities = random.sample(cities, len(cities))
        # finish cycle on the first city from the route
        self.sequence_of_cities.append(self.sequence_of_cities[0])

        self.calculate_route_length()

    def calculate_route_length(self):
        self.length = 0.0
        i = 0

        while i < len(self.sequence_of_cities) - 1:
            remote_city = self.sequence_of_cities[i + 1]
            distance = self.sequence_of_cities[i].distances[remote_city.name]

            self.length += distance
            i += 1