import argparse
import numpy as np
import matplotlib.pyplot as plt
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


def unzip_list_of_points(points):
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

    args = parser.parse_args()

    cities_number = args.cities_number
    cities_distribution = args.cities_distribution
    population_size = args.population_size

    cities = []

    coordinates_generator = CoordinatesGenerator(cities_number)

    if cities_distribution == "ud":
        points = coordinates_generator.uniform_distribution()
    elif cities_distribution == "cd":
        points = coordinates_generator.clustered_distribution()
    elif cities_distribution == "rd":
        points = coordinates_generator.random_distribution()
    else:
        print("Wrong distribution type")
        exit(1)

    for point in points:
        cities.append(City(point[COORD_X], point[COORD_Y], f"C{len(cities) + 1}"))

    for city in cities:
        city.fill_distances_dict(cities)

    population = Population(population_size, cities)
    print("Shortest path: " + str(population.get_fittest_route().length))

    plt.scatter(*unzip_list_of_points(points))
    plt.title(
        f"Cities arranged with {DISTRIBUTION_NAME[cities_distribution]} distribution"
    )
    plt.show()


if __name__ == "__main__":
    main()
