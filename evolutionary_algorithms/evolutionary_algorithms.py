import argparse
import numpy as np
import matplotlib.pyplot as plt
from city import City
from route import Route
from coordinates_generator import CoordinatesGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cn", "--cities_number", type=int, help="Number of cities")
    parser.add_argument(
        "-cd",
        "--cities_distribution",
        type=str,
        help="Type of cities distribution: [ud|cd|rd] - [uniform|clustered|random]",
    )

    args = parser.parse_args()

    cities_number = args.cities_number
    cities_distribution = args.cities_distribution

    cities = []

    coordinates_generator = CoordinatesGenerator(cities_number)

    if cities_distribution == "ud":
        X, Y = coordinates_generator.uniform_distribution()
        print(X)
    elif cities_distribution == "cd":
        X, Y = coordinates_generator.clustered_distribution()
    elif cities_distribution == "rd":
        X, Y = coordinates_generator.random_distribution()
        for i in range(cities_number):
            cities.append(City(X[i], Y[i], f"C{len(cities) + 1}"))
    else:
        print("Wrong distribution type")
        exit(1)

    for city in cities:
        city.fill_distances_dict(cities)

    route = Route(cities)

    plt.plot(X, Y, "-o")
    plt.show()


if __name__ == "__main__":
    main()
