import utils
import numpy as np
import math

class NaiveBayes:
    def __init__(self) -> None:
        self.gaussian_params = []
        self.discrete_params = []

    def gaussian_likelihood(self, data: float, mean: float, var: float) -> float:
        var += 1e-9
        coeff = 1 / math.sqrt(2 * math.pi * var)
        exponent = math.exp(-((data - mean) ** 2 / (2 * var)))
        return coeff * exponent

    def fit(self, features: list, labels: list, feature_types: list) -> None:
        "Fits the model based on the feature type (C or D)."
        self.labels = labels
        self.unique_labels = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        enumerated_labels = [self.unique_labels[label] for label in self.labels]
        self.res = list(dict.fromkeys(enumerated_labels))

        self.feature_types = feature_types

        for label in self.res:
            print(f"Label: {label}")
            label_features = [f for f, lab in zip(features, enumerated_labels) if lab == label]
            label_features_T = utils.transpose_array(label_features)

            gaussian_params = []
            discrete_params = []
            
            for i, col in enumerate(label_features_T):
                if (feature_types[i] == "C"):
                    updated_vals = []
                    for value in col:
                        updated_vals.append(float(value))
                    mean = utils.calc_mean(updated_vals)
                    var = utils.calc_var(updated_vals)
                    gaussian_params.append((mean, var))
                    print(f"  Feature {i} (continuous):")
                    print(f"    Mean: {mean:.3f}")
                    print(f"    Variance: {var:.3f}")
                else:
                    unique_vals, counts = np.unique(col, return_counts=True)
                    probabilities = {val: count / len(col) for val, count in zip(unique_vals, counts)}
                    discrete_params.append(probabilities)
                    print(f"  Feature {i} (discrete):")
                    for val, prob in probabilities.items():
                        print(f"    Value: {val}, Probability: {prob:.3f}")
            self.gaussian_params.append(gaussian_params)
            self.discrete_params.append(discrete_params)

    def predict(self, features: list) -> list:
        """
        Predicts the labels for the given features.
        Gaussian for continuous features and discrete probabilities for categorical features.
        """
        predictions = []
        for feature in features:
            label_probabilities = {}

            for label_idx, label in enumerate(self.res):
                label_count = sum(1 for l in self.labels if self.unique_labels[l] == label)
                prior = math.log(label_count / len(self.labels))
                likelihood = 0

                discrete_idx = 0
                for i, value in enumerate(feature):
                    if self.feature_types[i] == "C":
                        mean, var = self.gaussian_params[label_idx][i]
                        likelihood_val = self.gaussian_likelihood(float(value), mean, var)
                        likelihood += math.log(likelihood_val)
                    else:
                        feature_probs = self.discrete_params[label_idx][discrete_idx]
                        if value in feature_probs:
                            prob = feature_probs[value]
                        else:
                            print(f"\nApplied La place smoothing for label {label} and Value: {value} in feature: {i}")
                            total_count = sum(feature_probs.values())
                            num_unique = len(feature_probs)
                            alpha = 1.0
                            prob = alpha / (total_count + alpha * num_unique)
                        likelihood += math.log(prob)
                        discrete_idx += 1

                label_probabilities[label] = prior + likelihood
            print("\nProbabilities for each label:")
            print(label_probabilities)
            predictions.append(max(label_probabilities.items(), key=lambda item: item[1])[0])
        return predictions

if __name__ == "__main__":
    data = utils.csv_reader("./datasets/artificial_dataset.csv")
    
    clf = NaiveBayes()
    clf.fit(data.features, data.labels, data.categoryType)
    result = clf.predict([[38.0, "True", "True", "Little", "False"]])

    print("\nLabels:")
    for label, idx in clf.unique_labels.items():
        print(f"  {label}: {idx}")

    print(f"\nPredicted label index: {result[0]}")
    predicted_label_name = next(label for label, idx in clf.unique_labels.items() if idx == result[0])
    print(f"Predicted label name: {predicted_label_name}")
