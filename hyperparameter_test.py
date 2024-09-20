import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayesClassifier

def test_naive_bayes(train_data, train_labels, test_data, test_labels, alphas):
    accuracies = []
    for alpha in alphas:
        nb = NaiveBayesClassifier(alpha=alpha)
        nb.fit(train_data, train_labels)
        predictions = nb.predict(test_data)
        accuracy = sum([1 for pred, actual in zip(predictions, test_labels) if pred == actual]) / len(test_labels)
        accuracies.append(accuracy)
    plt.plot(alphas, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Accuracy vs Alpha')
    plt.show()

def test_logistic_regression(train_data, train_labels, test_data, test_labels, learning_rates, epochs=100):
    accuracies = []
    for lr in learning_rates:
        lr_model = LogisticRegression(learning_rate=lr, epochs=epochs)
        lr_model.fit(train_data, train_labels)
        predictions = lr_model.predict(test_data)
        accuracy = sum([1 for pred, actual in zip(predictions, test_labels) if pred == actual]) / len(test_labels)
        accuracies.append(accuracy)
    plt.plot(learning_rates, accuracies)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression Accuracy vs Learning Rate')
    plt.show()

# Example usage for testing:
# alphas = [0.1, 0.5, 1.0, 10, 100]
# test_naive_bayes(train_data, train_labels, test_data, test_labels, alphas)

# learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
# test_logistic_regression(train_data, train_labels, test_data, test_labels, learning_rates)
