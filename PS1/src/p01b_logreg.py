import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Build model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # Evaluate data
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    # Save predictions
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        theta = np.zeros(n)
        while True: 
            # Calculate Hessian and Gradient
            g = 1/(1 + np.exp(-x @ theta))
            H = ((x.T * g * (1 - g)) @ x)/m
            grad = (x.T @ (g - y))/m
            
            # Update theta via newton method
            theta_old = np.copy(theta)
            theta = theta - np.linalg.inv(H) @ grad
            # End if convergence
            if (np.linalg.norm(theta-theta_old, 1) < self.eps):
                break
        # Update models theta
        self.theta = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y_pred = 1/(1 + np.exp(-x @ self.theta))
        return y_pred
        # *** END CODE HERE ***