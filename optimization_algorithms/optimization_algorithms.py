import argparse
from enum import Enum
from timeit import timeit
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class OutcomeType(Enum):
    FOUND_MINIMUM = 1
    EPSILON_LIMIT = 2
    ITERATIONS_LIMIT = 3


class Result:
    def __init__(
        self,
        x: float,
        y: float,
        iterations: int,
        algorithm: str,
        outcome_type: OutcomeType,
        x_values: List[float] = None,
        y_values: List[float] = None,
    ):
        self.x = x
        self.y = y
        self.iterations = iterations
        self.algorithm = algorithm
        self.outcome_type = outcome_type
        self.x_values = x_values
        self.y_values = y_values

    def __str__(self):
        if self.outcome_type == OutcomeType.FOUND_MINIMUM:
            return f"{self.algorithm} found minimum after {self.iterations} iterations"
        if self.outcome_type == OutcomeType.EPSILON_LIMIT:
            return f"{self.algorithm} found solution with MSE lower than epsilon at point (x,y) = {(self.x, self.y)} after {self.iterations} iterations"
        if self.outcome_type == OutcomeType.ITERATIONS_LIMIT:
            return f"{self.algorithm} couldn't find solution in {self.iterations} iterations. Algorithm ended at (x, y) = {(self.x, self.y)} "


def rosenbrock_function(x: float, y: float):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_gradient(x: float, y: float):
    return np.array([400 * x ** 3 - 400 * x * y + 2 * x - 2, 200 * y - 200 * x ** 2])


def rosenbrock_hess(x: float, y: float):
    return np.matrix([[1200 * x ** 2 - 400 * y + 2, -400 * x], [-400 * x, 200]])


def gradient_update_x_and_y(x: float, y: float, beta: float):
    # calculate partial derivatives of rosenbrock function
    (dx, dy) = rosenbrock_gradient(x, y)

    # update value of x and y basing on calculated derivatives and given beta coefficient
    new_x = x - beta * dx
    new_y = y - beta * dy

    return (new_x, new_y)


def newton_update_x_and_y(x: float, y: float, beta: float):
    # calculate product of inverted hessian matrix and gradient of rosenbrock function
    # (author's note - i know it could be named better than dx/dy like partial derivatives, but i don't really have idea how)
    dx_dy = np.array(np.linalg.inv(rosenbrock_hess(x, y)) @ rosenbrock_gradient(x, y))
    (dx, dy) = (dx_dy[0][0], dx_dy[0][1])

    # update value of x and y basing on calculated derivatives and given beta coefficient
    new_x = x - beta * dx
    new_y = y - beta * dy

    return (new_x, new_y)


def gradient_descent(
    init_x: float,
    init_y: float,
    beta: float,
    epsilon: float,
    epochs: int,
    plot: bool = False,
    debug: bool = False,
) -> Result:
    (previous_x, previous_y) = (init_x, init_y)
    x_values = [init_x]
    y_values = [init_y]

    for e in range(epochs):
        (x, y) = gradient_update_x_and_y(previous_x, previous_y, beta)
        if plot == True:
            x_values.append(x)
            y_values.append(y)

        if debug == True:
            print(
                f"epoch: {e}, (x, y): {(x,y)}, MSE: {(0 - rosenbrock_function(x, y)) ** 2}"
            )

        if rosenbrock_function(x, y) == 0:
            return Result(
                x,
                y,
                e + 1,
                "Gradient descent",
                OutcomeType.FOUND_MINIMUM,
                x_values,
                y_values,
            )
        if rosenbrock_function(x, y) ** 2 < epsilon:
            return Result(
                x,
                y,
                e + 1,
                "Gradient descent",
                OutcomeType.EPSILON_LIMIT,
                x_values,
                y_values,
            )
        if e == epochs - 1:
            return Result(
                x,
                y,
                e + 1,
                "Gradient descent",
                OutcomeType.ITERATIONS_LIMIT,
                x_values,
                y_values,
            )

        (previous_x, previous_y) = (x, y)


def newton_method(
    init_x: float,
    init_y: float,
    beta: float,
    epsilon: float,
    epochs: int,
    plot: bool = False,
    debug: bool = False,
) -> Result:
    (previous_x, previous_y) = (init_x, init_y)
    x_values = [init_x]
    y_values = [init_y]

    for e in range(epochs):
        (x, y) = newton_update_x_and_y(previous_x, previous_y, beta)
        if plot == True:
            x_values.append(x)
            y_values.append(y)

        if debug == True:
            print(
                f"epoch: {e}, (x, y): {(x,y)}, MSE: {(0 - rosenbrock_function(x, y)) ** 2}"
            )

        if rosenbrock_function(x, y) == 0:
            return Result(
                x,
                y,
                e + 1,
                "Newton's method",
                OutcomeType.FOUND_MINIMUM,
                x_values,
                y_values,
            )
        if rosenbrock_function(x, y) ** 2 < epsilon:
            return Result(
                x,
                y,
                e + 1,
                "Newton's method",
                OutcomeType.EPSILON_LIMIT,
                x_values,
                y_values,
            )
        if e == epochs - 1:
            return Result(
                x,
                y,
                e + 1,
                "Newton's method",
                OutcomeType.ITERATIONS_LIMIT,
                x_values,
                y_values,
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
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Determines whether programme should plot (x, y) points or not",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Determines whether programme should print points calculated in each epoch",
    )

    args = parser.parse_args()

    init_x = args.init_x
    init_y = args.init_y
    beta = args.beta
    epsilon = args.epsilon
    iterations = args.iterations
    algorithm = args.algorithm
    plot = args.plot
    debug = args.debug

    if algorithm == "gd":
        # get results of one execution of gradient descent for given arguments
        result = gradient_descent(
            init_x, init_y, beta, epsilon, iterations, plot, debug
        )
        # calculate mean time of 10 executions of gradient descent for given arguments
        t = timeit(
            "gradient_descent(init_x, init_y, beta, epsilon, iterations)",
            "from __main__ import gradient_descent;"
            f"init_x = {init_x}; init_y = {init_y}; beta = {beta};"
            f"epsilon = {epsilon}; iterations = {iterations}",
            number=10,
        )
    elif algorithm == "nm":
        # get results of one execution of Newton's method for given arguments
        result = newton_method(init_x, init_y, beta, epsilon, iterations, plot, debug)
        # calculate mean time of 10 executions of Newton's method for given arguments
        t = timeit(
            "newton_method(init_x, init_y, beta, epsilon, iterations)",
            "from __main__ import newton_method;"
            f"init_x = {init_x}; init_y = {init_y}; beta = {beta};"
            f"epsilon = {epsilon}; iterations = {iterations}",
            number=10,
        )
    else:
        print("Wrong algorithm")
        exit(1)

    print(result)
    print(f"Mean time of calculations for 10 runs: {t} seconds")

    # plotting all calculated (x, y) coordinates and actual minimum of rosenbrock function
    if plot == True:
        X = np.arange(-13, 13, 0.1)
        Y = np.arange(-13, 13, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = rosenbrock_function(X, Y)

        # prepare contour plot
        plt.figure(figsize=(15, 10))
        plt.contour(X, Y, Z, 200)
        plt.xlim([-13, 13])
        plt.ylim([-13, 13])

        plt.plot(
            result.x_values,
            result.y_values,
            "--o",
            linewidth=0.7,
            markersize=2,
            color="darkblue",
            markerfacecolor="darkblue",
        )
        plt.plot(init_x, init_y, marker="x", markersize=10, color="g")
        plt.plot(1, 1, marker="x", markersize=10, color="r")
        plt.legend(
            ["Calculated points", "Initial point", "Rosenbrock function's minimum"]
        )
        plt.title(f"Series of points calculated by {result.algorithm} algorithm")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


if __name__ == "__main__":
    main()
