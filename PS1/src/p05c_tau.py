import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Load data
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # Build model
    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)

    # Search tau_values for the best tau (lowest MSE on the validation set)
    k = len(tau_values) 
    MSE = np.zeros(k)
    for i in range(k):

        # Fit a LWR model with the best tau value
        tau = tau_values[i]
        model.tau = tau
        y_pred_valid = model.predict(x_valid)
        MSE[i] = np.mean((y_pred_valid - y_valid)**2)
        print(f'valid set: tau={tau}, MSE={MSE[i]}')

        # Plot validation predictions on top of training set
        plt.figure()
        plt.plot(x_train, y_train, 'bx', linewidth=0.2)
        plt.plot(x_valid, y_pred_valid, 'ro', linewidth=0.2)
        plt.xlabel('x')
        plt.xlabel('y')
        plt.savefig(f'output/p05c_tau_{tau_values[i]}.png')
    
    # Choose best model
    tau_final = tau_values[np.argmin(MSE)]
    model.tau = tau_final

    # Run on the test set to get the MSE value
    y_pred_test = model.predict(x_test)
    MSE_test = np.mean((y_pred_test - y_test)**2)
    print(f'test set: tau={tau_final}, MSE={MSE_test}')  


    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred_test)
    
    # *** END CODE HERE ***
