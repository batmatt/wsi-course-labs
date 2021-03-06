import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--init-x", type=float, help="x value algorithm starts with")
parser.add_argument("-y", "--init-y", type=float, help="y value algorithm starts with")
parser.add_argument("-b", "--beta", type=float, help="Beta coefficient in both algorithms")
parser.add_argument("-e", "--epsilon", type=float, help="Stop condition, when changes between epochs become too small")
parser.add_argument("-i", "--iterations", type=int, help="Number of epochs")
parser.add_argument("-a", "--algorithm", type=str, help="Algorithm used to find minimum: [gd|nm]")

args = parser.parse_args()

init_x = args.init_x
init_y = args.init_y
beta = args.beta
epsilon = args.epsilon
iterations = args.iterations
algorithm = args.algorithm

def rosenbrock_function(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_gradient(x, y):
    return np.array([400*x**3 + 2*x - 400*x*y - 2,
     200*y - 200*x**2])

def rosenbrock_hess(x, y):
    return np.matrix([[1200*x**2 - 400*y + 2, -400*x],
    [-400*x, 200]])

def gradient_update_x_and_y(x, y, beta):
    (dx, dy) = rosenbrock_gradient(x, y)

    x = x - beta*dx
    y = y - beta*dy

    return (x, y)

def newton_update_x_and_y(x, y, beta):
    dx_dy = np.array(np.linalg.inv(rosenbrock_hess(x, y)) @ rosenbrock_gradient(x, y))
    (dx, dy) = (dx_dy[0][0], dx_dy[0][1])

    x = x - beta*dx
    y = y - beta*dy

    return (x, y)

def gradient_descent(initial_x, initial_y, beta, epsilon, epochs):
    (previous_x, previous_y) = (initial_x, initial_y)

    for e in range(epochs):
        (x, y) = gradient_update_x_and_y(previous_x, previous_y, beta)
        print(f"epoch: {e}, (x, y): {(x,y)}")
        if rosenbrock_function(x, y) == 0:
            print(f"Gradient descent: Found minimum at (x,y)={(x,y)} after {e+1} iterations")
            break
        if rosenbrock_function(x, y)**2 < epsilon:
            print(f"Gradient descent: MSE lower than {epsilon} after {e+1} iterations")
            break
        if e == epochs-1:
            print(f"Gradient descent: Couldn't find minimum after {e+1} iterations")

        (previous_x, previous_y) = (x, y)

def newton_method(initial_x, initial_y, beta, epsilon, epochs):
    (previous_x, previous_y) = (initial_x, initial_y)

    for e in range(epochs):
        (x, y) = newton_update_x_and_y(previous_x, previous_y, beta)
        print(f"epoch: {e}, (x, y): {(x,y)}")
        if rosenbrock_function(x, y) == 0:
            print(f"Newton's method: Found minimum at (x,y)={(x,y)} after {e+1} iterations")
            break
        if rosenbrock_function(x, y)**2 < epsilon:
            print(f"Newton's method: MSE lower than {epsilon} after {e+1} iterations")
            break
        if e == epochs-1:
            print(f"Newton's method: Couldn't find minimum after {e+1} iterations")

        (previous_x, previous_y) = (x, y)

def main():
    if args.algorithm == "gd":
        gradient_descent(init_x, init_y, beta, epsilon, iterations)
    if args.algorithm == "nm":
        newton_method(init_x, init_y, beta, epsilon, iterations)

if __name__ == "__main__":
    main()
