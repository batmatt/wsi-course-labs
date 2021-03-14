import numpy as np
import matplotlib.pyplot as plt
from city import City

cities = []


def main():
    c1 = City(0, 0, cities)
    c2 = City(5, 0, cities)
    c3 = City(0, 5, cities)
    c4 = City(5, 5, cities)

    for city in cities:
        city.fill_distances_dict(cities)
    print(cities[2].distances)


if __name__ == "__main__":
    main()
