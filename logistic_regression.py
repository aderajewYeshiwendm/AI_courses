import math


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        self.weights = [0] * len(X[0])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                linear_model = sum([self.weights[j] * X[i][j] for j in range(len(X[0]))]) + self.bias
                y_pred = self.sigmoid(linear_model)
                error = y[i] - y_pred
                self.weights = [self.weights[j] + self.learning_rate * error * X[i][j] for j in range(len(X[0]))]
                self.bias += self.learning_rate * error

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            linear_model = sum([self.weights[j] * X[i][j] for j in range(len(X[0]))]) + self.bias
            y_pred.append(1 if self.sigmoid(linear_model) > 0.5 else 0)
        return y_pred

# Example usage:
# lr = LogisticRegression(learning_rate=0.01, epochs=1000)
# lr.fit(train_data, train_labels)
# predictions = lr.predict(test_data)
