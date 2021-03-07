import argparse
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class OutcomeType(Enum):
    FOUND_MINIMUM = 1
    EPSILON_LIMIT = 2
    ITERATIONS_LIMIT = 3


class Result:
    def __init__(
        self,
        initial_x: float,
        initial_y: float,
        iterations: int,
        algorithm: str,
        outcome_type: OutcomeType,
    ):
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.iterations = iterations
        self.algorithm = algorithm
        self.outcome_type = outcome_type

    def __str__(self):
        if self.outcome_type == OutcomeType.FOUND_MINIMUM:
            return f"{self.algorithm} found minimum after {self.iterations} iterations"
        if self.outcome_type == OutcomeType.EPSILON_LIMIT:
            return f"{self.algorithm} found solution's square lower than epsilon after {self.iterations} iterations"
        if self.outcome_type == OutcomeType.ITERATIONS_LIMIT:
            return f"{self.algorithm} couldn't find solution in {self.iterations} iterations"


def rosenbrock_function(x: float, y: float):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(x: float, y: float):
    return np.array([400 * x ** 3 + 2 * x - 400 * x * y - 2, 200 * y - 200 * x ** 2])


def rosenbrock_hess(x: float, y: float):
    return np.matrix([[1200 * x ** 2 - 400 * y + 2, -400 * x], [-400 * x, 200]])


def gradient_update_x_and_y(x: float, y: float, beta: float):
    (dx, dy) = rosenbrock_gradient(x, y)

    new_x = x - beta * dx
    new_y = y - beta * dy

    return (new_x, new_y)


def newton_update_x_and_y(x: float, y: float, beta: float):
    dx_dy = np.array(np.linalg.inv(rosenbrock_hess(x, y)) @ rosenbrock_gradient(x, y))
    (dx, dy) = (dx_dy[0][0], dx_dy[0][1])

    new_x = x - beta * dx
    new_y = y - beta * dy

    return (new_x, new_y)


def gradient_descent(
    init_x: float, init_y: float, beta: float, epsilon: float, epochs: int
):
    (x, y) = (init_x, init_y)
    (previous_x, previous_y) = (init_x, init_y)

    for e in range(epochs):
        (x, y) = gradient_update_x_and_y(previous_x, previous_y, beta)
        print(f"epoch: {e}, (x, y): {(x,y)}")
        if rosenbrock_function(x, y) == 0:
            return Result(
                init_x, init_y, e + 1, "Gradient descent", OutcomeType.FOUND_MINIMUM
            )
        if rosenbrock_function(x, y) ** 2 < epsilon:
            return Result(
                init_x, init_y, e + 1, "Gradient descent", OutcomeType.EPSILON_LIMIT
            )
        if e == epochs - 1:
            return Result(
                init_x, init_y, e + 1, "Gradient descent", OutcomeType.ITERATIONS_LIMIT
            )

        (previous_x, previous_y) = (x, y)


def newton_method(
    init_x: float, init_y: float, beta: float, epsilon: float, epochs: int
):
    (x, y) = (init_x, init_y)
    (previous_x, previous_y) = (init_x, init_y)

    for e in range(epochs):
        (x, y) = newton_update_x_and_y(previous_x, previous_y, beta)
        print(f"epoch: {e}, (x, y): {(x,y)}")
        if rosenbrock_function(x, y) == 0:
            return Result(
                init_x, init_y, e + 1, "Gradient descent", OutcomeType.FOUND_MINIMUM
            )
        if rosenbrock_function(x, y) ** 2 < epsilon:
            return Result(
                init_x, init_y, e + 1, "Gradient descent", OutcomeType.EPSILON_LIMIT
            )
        if e == epochs - 1:
            return Result(
                init_x, init_y, e + 1, "Gradient descent", OutcomeType.ITERATIONS_LIMIT
            )

        (previous_x, previous_y) = (x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--init-x", type=float, help="x value algorithm starts with"
    )
    parser.add_argument(
        "-y", "--init-y", type=float, help="y value algorithm starts with"
    )
    parser.add_argument(
        "-b", "--beta", type=float, help="Beta coefficient in both algorithms"
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        help="Stop condition, when changes between epochs become too small",
    )
    parser.add_argument("-i", "--iterations", type=int, help="Number of epochs")
    parser.add_argument(
        "-a", "--algorithm", type=str, help="Algorithm used to find minimum: [gd|nm]"
    )

    args = parser.parse_args()

    init_x = args.init_x
    init_y = args.init_y
    beta = args.beta
    epsilon = args.epsilon
    iterations = args.iterations
    algorithm = args.algorithm

    if args.algorithm == "gd":
        result = gradient_descent(init_x, init_y, beta, epsilon, iterations)
    if args.algorithm == "nm":
        result = newton_method(init_x, init_y, beta, epsilon, iterations)

    print(result)


if __name__ == "__main__":
    main()
