import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Load data
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    _, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)


    # Part (c): Train and test on true labels
    # Build model
    t_model  = LogisticRegression()
    t_model.fit(x_train, t_train)

    # Make sure to save outputs to pred_path_c
    t_pred = t_model.predict(x_test)
    np.savetxt(pred_path_c, t_pred > 0.5, fmt='%d')
    util.plot(x_test, t_test, t_model.theta, 'output/p02c.png'.format(pred_path[-5])) 

    # Part (d): Train on y-labels and test on true labels
    # Build model
    y_model  = LogisticRegression()
    y_model.fit(x_train, y_train)

    # Make sure to save outputs to pred_path_d
    y_pred = y_model.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')
    util.plot(x_test, t_test, y_model.theta, 'output/p02d.png'.format(pred_path[-5])) 

    # Part (e): Apply correction factor using validation set and test on true labels
    # Calculate alpha
    h_x_valid = y_model.predict(x_valid)
    alpha = np.sum(h_x_valid) / x_valid.shape[0]

    # Correct the d) results
    corrected_pred = y_pred / alpha 

    # Plot and use np.savetxt to save outputs to pred_path_e
    np.savetxt(pred_path_e, corrected_pred > 0.5, fmt='%d')
    correction = 1 + np.log(2 / alpha - 1) / y_model.theta[0] # See theory solutions
    util.plot(x_test, t_test, y_model.theta, 'output/p02e.png'.format(pred_path[-5]), correction=correction) 
    # *** END CODER HERE
