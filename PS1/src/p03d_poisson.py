import numpy as np
import util
from matplotlib import pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Load data
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # Fit a Poisson Regression model
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred)

    # Plot to check answers
    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('true counts')
    plt.ylabel('predict counts')
    plt.savefig('output/p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # Initialize
        m, n = x.shape
        self.theta = np.zeros(n)
        while True:
            theta = np.copy(self.theta)
            """ SGD doent converge
            for i in range(m):
                self.theta = self.theta + self.step_size * (y[i] - np.exp(self.theta.T @ x[i])) * x[i]
            """
            # Batch GD does converge and works perfectly
            self.theta += self.step_size * x.T.dot(y - np.exp(x.dot(self.theta))) / m

            # Untill convergeance
            if np.linalg.norm(self.theta - theta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***
