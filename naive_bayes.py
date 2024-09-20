from collections import defaultdict
import math

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_freqs = None
        self.feature_freqs = None

    def fit(self, X, y):
        self.classes = set(y)
        self.class_freqs = defaultdict(int)
        self.feature_freqs = defaultdict(lambda: defaultdict(int))
        self.total_samples = len(y)

        for features, label in zip(X, y):
            self.class_freqs[label] += 1
            for feature in features:
                self.feature_freqs[label][feature] += 1

    def predict(self, X):
        predictions = []
        for features in X:
            class_probs = {}
            for cls in self.classes:
                class_probs[cls] = math.log(self.class_freqs[cls] / self.total_samples)
                for feature in features:
                    feature_count = self.feature_freqs[cls][feature]
                    class_probs[cls] += math.log((feature_count + self.alpha) / (self.class_freqs[cls] + self.alpha * len(self.feature_freqs[cls])))
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

# Example usage:
# nb = NaiveBayesClassifier(alpha=1.0)
# nb.fit(train_data, train_labels)
# predictions = nb.predict(test_data)
