from route import Route


class Population:
    def __init__(self, is_initial=False, population_size=0, cities=[]):
        self.routes_population = []

        if is_initial:
            for i in range(population_size):
                self.routes_population.append(Route(cities))

    def get_fittest_route(self):
        """
        self --> Route

        Sorts routes_population by their length and returns
        the shortest one

        """
        sorted_routes = sorted(self.routes_population, key=lambda route: route.length)
        return sorted_routes[0]