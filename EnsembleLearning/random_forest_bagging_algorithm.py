import numpy as np
from scipy import stats

from random_forest_decision_tree_algorithm import ID3DecisonTree

class BaggingAlgorithm:
    def __init__(self, num_iterations, sample_size):
        self.num_iterations = num_iterations
        self.sample_size = sample_size
    
    def fit(self, data_set, size_attribute_subset):
        n_samples = len(data_set)
        trees = []
        
        for i in range(self.num_iterations):
            index = np.random.choice(n_samples, size=self.sample_size, replace=True)
            data = data_set.iloc[index]

            label_index = len(data.columns) - 1
            tree = ID3DecisonTree(dataset=data, 
                                label_index=label_index, 
                                max_depth=10, 
                                impurity_measure_metric="entropy").construct_tree(current_set=data, attributes=list(range(label_index)), size_attribute_subset=size_attribute_subset)

            trees.append(tree)
        return trees

    @staticmethod
    def get_mode(predictions):
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)] 

    @staticmethod
    def predict(data_set, num_tree, trees):
        tree_predictions = np.zeros((len(data_set), num_tree))
        for i, tree in enumerate(trees):
            tree_predictions[:, i] = np.array([ID3DecisonTree.predict_label(tree, data_set.iloc[j]) for j in range(len(data_set))])
        final_prediction =  np.apply_along_axis(BaggingAlgorithm.get_mode, axis=1, arr=tree_predictions)

        return final_prediction
    
    @staticmethod
    def return_prediction(data_set, tree):
        tree_prediction = np.array([ID3DecisonTree.predict_label(tree, data_set.iloc[j]) for j in range(len(data_set))])

        return tree_prediction
    
    @staticmethod
    def return_mode(tree_predictions):
        row_modes = stats.mode(tree_predictions, axis=1).mode
        row_modes = row_modes.flatten()
        return row_modes

    @staticmethod
    def calculate_error(y, prediction):
        error = np.mean(prediction != y)
        return error