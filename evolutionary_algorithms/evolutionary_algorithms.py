import argparse
import matplotlib.pyplot as plt
import random
import copy
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


def evolutionary_algorithm_loop(
    initial_population: Population,
    population_size: int,
    tournament_size: int,
    mutation_threshold: float,
    iterations_limit: int,
    plot: bool = False,
):
    """
    Function aggregating other methods into loop implementing main
    evolutionary algorithm logic.

    """
    # setting initial best route
    fittest_route = copy.deepcopy(initial_population.get_fittest_route())
    print(f"Fittest route of initial population:\n{fittest_route}")
    best_generation = 0

    # if plot option is enabled store lengths of best routes for each generation
    # in the end we plot length as a function of iteration
    if plot:
        fittest_routes_lengths = [fittest_route.length]

    generation = 1
    old_population = initial_population
    while generation < iterations_limit:
        new_population = Population()

        for x in range(population_size):
            child_route = tournament_selection(tournament_size, old_population)
            new_population.routes_population.append(child_route)

        mutate(mutation_threshold, new_population)

        if new_population.get_fittest_route().length < fittest_route.length:
            fittest_route = copy.deepcopy(new_population.get_fittest_route())
            best_generation = generation

        if plot:
            fittest_routes_lengths.append(fittest_route.length)

        # if plot option is enabled plot population's best route every 1/5 of iterations_limit
        if plot and (
            generation % (0.2 * iterations_limit) == 0
            or generation == iterations_limit - 1
        ):
            X = []
            Y = []
            for city in fittest_route.sequence_of_cities:
                X.append(city.x)
                Y.append(city.y)

            plt.plot(X[0], Y[0], marker="o", markersize=5, color="r")
            plt.plot(X, Y, "--o")
            plt.title(
                f"Fittest sequence of cities in {generation} generation\nLength: {fittest_route.length}"
            )

            plt.legend(
                [
                    f"Starting city: {fittest_route.sequence_of_cities[0].name}",
                ]
            )
            plt.show()

        old_population = new_population
        generation = generation + 1

    print(
        f"Fittest route overall:\n{fittest_route}\nFound in {best_generation}. generation"
    )

    if plot:
        plt.plot(range(0, iterations_limit), fittest_routes_lengths)
        plt.title("Fittest route in each generation")
        plt.xlabel("generation")
        plt.ylabel("length of the fittest route")
        plt.show()


def tournament_selection(tournament_size: int, population: Population):
    """
    int, Population --> Route

    Creates tournament population containing tournament_size
    randomly selected routes from initial population and returns
    the fittest one.

    """
    tournament_population = Population()
    tournament_population.routes_population = random.sample(
        population.routes_population, tournament_size
    )

    return tournament_population.get_fittest_route()


def mutate(mutation_threshold: float, population: Population):
    """
    float, Population --> None

    Mutates population individuals by swapping two random
    cities in the random route.

    """

    mutations_number = 0
    mutation_population = Population()

    for route in population.routes_population:
        if random.random() > mutation_threshold:
            # pick subset of routes that will mutate
            mutations_number = mutations_number + 1
            mutation_population.routes_population = random.sample(
                population.routes_population, 1
            )

    for route in mutation_population.routes_population:
        # randomize indices of two cities in chosen route
        # skip first and last element, because it could break route
        first_city_index, second_city_index = random.sample(
            range(1, len(route.sequence_of_cities) - 1), 2
        )
        # swap cities at two random indices using Python swap idiom
        (
            route.sequence_of_cities[first_city_index],
            route.sequence_of_cities[second_city_index],
        ) = (
            route.sequence_of_cities[second_city_index],
            route.sequence_of_cities[first_city_index],
        )
        # calculate length of route after mutation
        route.length = route.calculate_route_length()


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
    parser.add_argument(
        "-plt",
        "--plot",
        action="store_true",
        help="Enables plotting options",
    )

    args = parser.parse_args()

    _cities_number = args.cities_number
    _cities_distribution = args.cities_distribution
    _population_size = args.population_size
    _tournament_size = args.tournament_size
    _mutation_threhold = args.mutation_threshold
    _iterations = args.iterations
    _plot = args.plot

    _cities = []

    _coordinates_generator = CoordinatesGenerator(_cities_number)

    if _cities_distribution == "ud":
        points = _coordinates_generator.uniform_distribution()
    elif _cities_distribution == "cd":
        points = _coordinates_generator.clustered_distribution()
    elif _cities_distribution == "rd":
        points = _coordinates_generator.random_distribution()
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

    _initial_population = Population(
        is_initial=True, population_size=_population_size, cities=_cities
    )

    evolutionary_algorithm_loop(
        _initial_population,
        _population_size,
        _tournament_size,
        _mutation_threhold,
        _iterations,
        _plot,
    )


if __name__ == "__main__":
    main()
