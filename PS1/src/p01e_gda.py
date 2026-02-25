import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # Not intercept yet, as GDA builds mle without it.
    
    # *** START CODE HERE ***
    # Build model
    GDAmodel = GDA()
    GDAmodel.fit(x_train, y_train)

    # Evaluate data
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    y_pred = GDAmodel.predict(x_valid)
    util.plot(x_train, y_train, GDAmodel.theta, 'output/p01e_{}.png'.format(pred_path[-5])) 
    # Save predictions
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        # Mles
        fi = np.sum(y) / m
        mu_0 = np.sum(x[y==0], axis=0)/np.sum(y==0) 
        mu_1 = np.sum(x[y==1], axis=0)/np.sum(y==1) 
        sigma = ((x[y==0] - mu_0).T @  (x[y==0] - mu_0) + (x[y==1] - mu_1).T @  (x[y==1] - mu_1))/m
      
        # Get theta from posteriori distribution
        self.theta[0] = 1/2 * (mu_0 + mu_1).T @ np.linalg.inv(sigma) @ (mu_0 - mu_1) - np.log((1-fi)/fi)
        self.theta[1:] = np.linalg.inv(sigma) @ (mu_1 - mu_0)
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        y_pred = 1/(1+np.exp(-x @ self.theta))
        return y_pred
        # *** END CODE HERE
