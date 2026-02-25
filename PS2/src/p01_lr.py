# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y, a_or_b):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break

        # See evolution of theta
        """
        if i % 100000 == 0:
            util.plot(X, (Y == 1).astype(int), theta, save_path=f'output/{a_or_b}-{i}')
            print(f'iteration:{i}, theta:{theta} difference:{np.linalg.norm(prev_theta - theta)}')
        """
        #end

        # We can see how the separating hyperplane isnt changing along iterations, so we have found an stable solution,
        # however, we dont see convergence in dataset B, because even though the difference between theta and previous theta
        # is decreasing, their values keep increasing with each iteration, so the difference doesn`t approach the e-15 tol.
        # This increase is due to the logistic loss with y in -1, 1 beign vulnerable to scaling in dataset B, since exp(-y*theta*x)
        # keps decreasing with an increase in ||theta|| when data is linearly separable (y*theta*x > 0 for all examples)

    return


def main():

    # Investigate training difference
    from matplotlib import pyplot as plt
    from util import plot_points

    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=False)
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=False)

    plt.figure()
    plot_points(Xa, (Ya==1).astype(int))
    plt.savefig('output/p01a.png')

    plt.figure()
    plot_points(Xb, (Yb==1).astype(int))
    plt.savefig('output/p01b.png')
    #end

    # We can see how dataset B is linearly separable, while A isn`t.

    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya, 'a')

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb, 'b')


if __name__ == '__main__':
    main()
