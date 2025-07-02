import warnings
import numpy as np
import time
import pandas as pd
import sys
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, root_mean_squared_error,
    mean_squared_error, mean_absolute_percentage_error, r2_score, f1_score, hamming_loss, explained_variance_score,mean_absolute_percentage_error, median_absolute_error, 
)
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
    
    start = time.time()

    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Generate predictions using the model
    predictions = model.predict(X_test)
    end = time.time()
    time_taken = end - start
    y_test = np.array(y_test)

    # The first column is the current stock price placeholder
    # We want to compare each remaining column's predicted and actual values relative to the first column.

    # Calculate direction of movement for predictions relative to first column
    # 1 if went up or stayed same, 0 if went down
    pred_direction = (predictions[:, 1:] - predictions[:, [0]]) >= 0
    binary_predictions = pred_direction.astype(int)

    # Similarly, calculate direction for actual y_test relative to first column
    actual_direction = (y_test[:, 1:] - y_test[:, [0]]) >= 0
    y_test_binary = actual_direction.astype(int)

    # Now compute balanced accuracy for each output column and average it
    balanced_accuracies = 0
    for i in range(y_test_binary.shape[1]):
        balanced_acc = balanced_accuracy_score(y_test_binary[:, i], binary_predictions[:, i])
        balanced_accuracies += balanced_acc
    mba = balanced_accuracies / y_test_binary.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # For overall accuracy, hamming loss, precision, recall, f1 - compare binary labels
        acc = accuracy_score(y_test_binary, binary_predictions)
        ham_acc = 1 - hamming_loss(y_test_binary, binary_predictions)
        precision = precision_score(y_test_binary, binary_predictions, average="weighted")
        recall = recall_score(y_test_binary, binary_predictions, average="weighted")
        f1 = f1_score(y_test_binary, binary_predictions, average="weighted")

        # For regression metrics like MAE, RMSE, R2 use the original values from columns 1 onward only
        mae = mean_absolute_error(y_test[:, 1:], predictions[:, 1:])
        rmse = root_mean_squared_error(y_test[:, 1:], predictions[:, 1:])
        mape = mean_absolute_percentage_error(y_test[:, 1:], predictions[:, 1:])
        r2 = r2_score(y_test[:, 1:], predictions[:, 1:])

    round_digits = 4
    results = [
        time_taken,
        round(rmse, round_digits),
        round(r2,round_digits),
        round(mape*100,6),
        round(acc, round_digits),
        round(mba, round_digits),
        round(ham_acc, round_digits),
        round(precision, round_digits),
        round(recall, round_digits),
        round(f1, round_digits)
    ]

    per_output_accuracies = []
    per_output_maes = []

    n_outputs = y_test_binary.shape[1]
    for i in range(n_outputs):
        output_acc = balanced_accuracy_score(y_test_binary[:, i], binary_predictions[:, i])
        per_output_accuracies.append(round(output_acc, round_digits))

        output_mae = mean_absolute_error(y_test[:, i+1], predictions[:, i+1])
        per_output_maes.append(round(output_mae, round_digits))

    results.extend(per_output_accuracies + per_output_maes)

    print(rmse)
    print("----------------------------------------")
    data_logger.save_info(model, "Full", results)
    return rmse