import numpy as np

from weighted_decision_tree_algorithm import WeightedID3DecisonTree


import numpy as np

class AdaBoostAlgorithm:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations
        self.trees = []
        self.alphas = []

    def fit(self, data_set: np.ndarray):
        n_samples = len(data_set)
        weights = np.ones(n_samples) / n_samples
        label_index=len(data_set.columns)-1
        
        for n_iter in range(self.num_iterations):
            tree = WeightedID3DecisonTree(data_set, 
                                        label_index=len(data_set.columns)-1,
                                        max_depth=1, 
                                        impurity_measure_metric="entropy").construct_tree(current_set=data_set, attributes = list(range(data_set.shape[1] - 1)), weights = weights)

            weighted_error = WeightedID3DecisonTree.calculate_error(tree, data_set, weights)

            alpha = (1/2) * np.log((1 - weighted_error) / weighted_error)
            self.trees.append(tree)
            self.alphas.append(alpha)

            for i in range(n_samples):
                prediction = WeightedID3DecisonTree.predict_label(tree, data_set.iloc[i])
                if prediction != data_set[label_index][i]:
                    weights[i] = weights[i] * np.exp(alpha)
                else:
                    weights[i] = weights[i] * np.exp(-alpha)

            weights = weights / np.sum(weights)

    def predict(self, data_set: np.ndarray):
        prediction = np.zeros(len(data_set))
        for alpha, classifier in zip(self.alphas, self.trees):
            stump_prediction = np.array([WeightedID3DecisonTree.predict_label(classifier, data_set.iloc[i]) for i in range(len(data_set))])
            prediction = prediction + alpha * stump_prediction

        return np.sign(prediction)

    def calculate_error(self, data_set: np.ndarray):
        predictions = self.predict(data_set)
        return np.mean(predictions != data_set.iloc[:, -1])