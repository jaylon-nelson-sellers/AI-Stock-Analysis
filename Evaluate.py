import warnings
import numpy as np
import time
import pandas as pd
import sys
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, root_mean_squared_error,
    mean_squared_error, mean_absolute_percentage_error, r2_score, f1_score, hamming_loss, explained_variance_score
)
from pympler import asizeof
def evaluate_model(model, X_train, X_test, y_train, y_test, data_logger):
    """
    Fits a model and evaluates it using specified metrics based on problem type (regression or classification).
    Also evaluates performance on specific groups of stocks.
    """

    import time
    import warnings
    import numpy as np
    from sklearn.metrics import (balanced_accuracy_score, accuracy_score, hamming_loss, 
                                 precision_score, recall_score, f1_score, mean_absolute_error,
                                 r2_score)
    
    # Assuming that there's a custom function for RMSE
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    start = time.time()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Generate predictions using the model
    predictions = model.predict(X_test)
    end = time.time()
    time_taken = end - start
    y_test = np.array(y_test)

    # Convert float predictions to binary (0 or 1)
    binary_predictions = np.where(predictions > 0, 1, 0)
    # Ensure y_test is also binary
    y_test_binary = np.where(y_test > 0, 1, 0)

    # Compute balanced accuracy for each output column and take the mean
    balanced_accuracies = 0
    for i in range(y_test_binary.shape[1]):
        balanced_acc = balanced_accuracy_score(y_test_binary[:, i], binary_predictions[:, i])
        balanced_accuracies += balanced_acc
    mba = balanced_accuracies / y_test_binary.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = accuracy_score(y_test_binary, binary_predictions)
        ham_acc = 1 - hamming_loss(y_test_binary, binary_predictions)
        precision = precision_score(y_test_binary, binary_predictions, average="weighted")
        recall = recall_score(y_test_binary, binary_predictions, average="weighted")
        f1 = f1_score(y_test_binary, binary_predictions, average="weighted")
        mae = mean_absolute_error(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    round_digits = 4
    # Initial set of overall results
    results = [
        time_taken,
        round(mae, round_digits),
        round(rmse, round_digits),
        round(r2, round_digits),
        round(acc, round_digits),
        round(mba, round_digits),
        round(ham_acc, round_digits),
        round(precision, round_digits),
        round(recall, round_digits),
        round(f1, round_digits)
    ]

    # Dynamically add per-output metrics:
    # First, accuracy for each output then mean absolute error for each output.
    per_output_accuracies = []
    per_output_maes = []
    
    # Loop over each output column. Assumes predictions and y_test have same shape.
    n_outputs = y_test_binary.shape[1]
    for i in range(n_outputs):
        # Compute accuracy for the i-th output using binary predictions
        output_acc = balanced_accuracy_score(y_test_binary[:, i], binary_predictions[:, i])
        per_output_accuracies.append(round(output_acc, round_digits))
        
        # Compute MAE for the i-th output using raw predictions
        output_mae = mean_absolute_error(y_test[:, i], predictions[:, i])
        per_output_maes.append(round(output_mae, round_digits))
    
    # Extend the results object with per-output accuracies first then maes.
    results.extend(per_output_accuracies + per_output_maes)

    # If you are logging data, consider adding the new columns to your data_logger here.
    # For example: data_logger.log_results(results)


    data_logger.save_info(model, "Full", results)
    return rmse