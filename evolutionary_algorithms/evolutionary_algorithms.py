import argparse
import matplotlib.pyplot as plt
import random
from city import City
from population import Population
from coordinates_generator import CoordinatesGenerator

COORD_X = 0
COORD_Y = 1
DISTRIBUTION_NAME = {
    "ud": "uniform",
    "cd": "clustered",
    "rd": "random",
}


def tournament_selection(tournament_size: int, population: Population):
    """
    int, Population --> Route

    Creates tournament population containing tournament_size
    randomly selected routes from initial population and returns
    the fittest one.

    """
    tournament_population = Population()

    for i in range(tournament_size):
        random_route = random.choice(population.routes_population)
        tournament_population.routes_population.append(random_route)

    return tournament_population.get_fittest_route()


def mutate(mutation_threshold: float, population: Population):
    """"""

    if random.random() > mutation_threshold:
        # pick random route from population
        random_route = random.choice(population.routes_population)

        print(f"Route before mutation: {random_route}")

        # randomize indices of two cities in chosen route
        first_city_index, second_city_index = random.sample(
            range(len(random_route.sequence_of_cities)), 2
        )

        print(first_city_index, second_city_index)

        # swap cities at two random indices using Python swap idiom
        (
            random_route.sequence_of_cities[first_city_index],
            random_route.sequence_of_cities[second_city_index],
        ) = (
            random_route.sequence_of_cities[second_city_index],
            random_route.sequence_of_cities[first_city_index],
        )

        # calculate length of route after mutation
        random_route.length = random_route.calculate_route_length()

        print(f"Route after mutation mutation: {random_route}")


def unzip_list_of_points(points):
    """
    Returns unzipped list of points, so it can be plotted
    with scatter function.

    """
    return list(zip(*points))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cn", "--cities_number", type=int, help="Number of cities")
    parser.add_argument(
        "-cd",
        "--cities_distribution",
        type=str,
        help="Type of cities distribution: [ud|cd|rd] - [uniform|clustered|random]",
    )
    parser.add_argument(
        "-p", "--population_size", type=int, help="Size of routes population"
    )
    parser.add_argument(
        "-t",
        "--tournament_size",
        type=int,
        help="Number of individuals taking part in tournament",
    )
    parser.add_argument(
        "-m",
        "--mutation_threshold",
        type=float,
        help="Threshold above which mutation occurs",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, help="Maximum number of iterations"
    )

    args = parser.parse_args()

    _cities_number = args.cities_number
    _cities_distribution = args.cities_distribution
    _population_size = args.population_size
    _tournament_size = args.tournament_size
    _mutation_threhold = args.mutation_threshold
    _iterations = args.iterations

    _cities = []

    coordinates_generator = CoordinatesGenerator(_cities_number)

    if _cities_distribution == "ud":
        points = coordinates_generator.uniform_distribution()
    elif _cities_distribution == "cd":
        points = coordinates_generator.clustered_distribution()
    elif _cities_distribution == "rd":
        points = coordinates_generator.random_distribution()
    else:
        print("Wrong distribution type")
        exit(1)

    for point in points:
        _cities.append(City(point[COORD_X], point[COORD_Y], f"C{len(_cities) + 1}"))

    for city in _cities:
        city.fill_distances_dict(_cities)

    plt.scatter(*unzip_list_of_points(points))
    plt.title(
        f"Cities arranged with {DISTRIBUTION_NAME[_cities_distribution]} distribution"
    )
    plt.show()

    _population = Population(
        is_initial=True, population_size=_population_size, cities=_cities
    )
    print(f"Fittest route of initial population:\n{_population.get_fittest_route()}")

    tournament_selection(_tournament_size, _population)
    mutate(_mutation_threhold, _population)


if __name__ == "__main__":
    main()
