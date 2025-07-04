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
    binary_predictions = model.predict(X_test)
    end = time.time()
    time_taken = end - start
    y_test_binary = np.array(y_test)

    

    # Now compute balanced accuracy for each output column and average it
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

    round_digits = 4
    # Initial set of overall results
    results = [
        time_taken,
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
        # Compute accuracy for the i-th output using binary predictions
        output_acc = balanced_accuracy_score(y_test_binary[:, i], binary_predictions[:, i])
        per_output_accuracies.append(round(output_acc, round_digits))
    
    # Extend the results object with per-output accuracies first then maes.
    results.extend(per_output_accuracies + per_output_maes)


    results.extend(per_output_maes)

    print(acc)
    print("----------------------------------------")
    data_logger.save_info(model, "Full", results)
    return acc