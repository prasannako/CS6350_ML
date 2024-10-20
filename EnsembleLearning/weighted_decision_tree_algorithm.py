import numpy as np


class ImpurityMeasureMetrics:
    @staticmethod
    def entropy(values: np.ndarray, weights) -> float:
        total_weight = np.sum(weights)
        categories, categories_counts = np.unique(values, return_counts=True)
        probabilities = [np.sum(weights[values == category])/total_weight for category in categories]
        entropy_ = -np.sum(p*np.log2(p) for p in probabilities if p>0)
        return entropy_

    
class WeightedID3DecisonTree():
    def __init__(self, dataset: np.ndarray, label_index: int, max_depth: int, impurity_measure_metric: str) -> None:
        self.dataset = dataset
        self.label_index = label_index
        self.max_depth = max_depth
        self.impurity_measure_metric = impurity_measure_metric
    
    def calculate_gain(self, data: np.ndarray, attribute: list, weights) -> float:
        impurity_function = {
            "entropy": ImpurityMeasureMetrics.entropy
        }.get(self.impurity_measure_metric)

        if impurity_function is None:
            raise ValueError("Use 'entropy'. ")

        before_split_impurity = impurity_function(data[attribute], weights)

        categories, categories_counts = np.unique(data[self.label_index], return_counts=True)

        after_split_impurity = 0
        total_weight = np.sum(weights)

        for i in range(len(categories)):
            subset = data[data[self.label_index] == categories[i]][attribute]
            subset_weights = weights[data[self.label_index] == categories[i]]
            impurity = impurity_function(subset, subset_weights)
            
            after_split_impurity += (np.sum(subset_weights) / total_weight)* impurity

        gain = before_split_impurity - after_split_impurity
        return gain
    
    def get_best_attribute_to_split(self, data: np.ndarray, attributes: list, weights) -> int:
        item_values = [self.calculate_gain(data, attribute, weights) for attribute in attributes]
        best_attribute_index = np.argmax(item_values)
        best_attribute = attributes[best_attribute_index]

        return best_attribute

    def remove_attribute(self, attributes: list, feature_to_remove: int) -> list:
        remaining_attributes = []
        for feature in attributes:
            if feature != feature_to_remove:
                remaining_attributes.append(feature)
        
        return remaining_attributes

    def get_majority_label(self, set: np.ndarray, weights):
        labels = set[self.label_index]
        categories, categories_counts = np.unique(labels, return_counts=True)
        majority_weight = np.argmax([np.sum(weights[labels == value]) for value in categories])
        return categories[majority_weight]

    def construct_tree(self, current_set: np.ndarray, attributes: list, weights: np.ndarray, current_depth: int = 0, tree: dict = None) -> dict:
        if len(np.unique(current_set[self.label_index])) <= 1:
            return np.unique(current_set[self.label_index])[0]
        
        elif len(attributes) == 0:
            return self.get_majority_label(current_set, weights)
        
        elif current_depth >= self.max_depth:
            return self.get_majority_label(current_set, weights)
        
        elif len(current_set) == 0:
            return self.get_majority_label(self.dataset, weights)
        
        else:
            current_depth += 1

            best_attribute_to_split = self.get_best_attribute_to_split(current_set, attributes, weights)

            tree = {best_attribute_to_split: {}}

            remaining_attributes = self.remove_attribute(attributes, best_attribute_to_split)
        
            for value in np.unique(current_set[best_attribute_to_split]):
                sub_set = current_set[current_set[best_attribute_to_split] == value]
                sub_weights = weights[current_set[best_attribute_to_split] == value]
                subtree = self.construct_tree(sub_set, remaining_attributes, sub_weights, current_depth, tree)
                tree[best_attribute_to_split][value] = subtree
            
            return tree

    @classmethod
    def predict_label(cls, tree, sample):
        current_node = tree
        while isinstance(current_node, dict): 
            attribute = next(iter(current_node)) 
            value = sample[attribute]
            
            if value in current_node[attribute]:
                current_node = current_node[attribute][value]
            else:
                return None
            
        return current_node 
    
    @classmethod
    def calculate_error(cls, tree, data, weights):
        correct_predictions = 0

        for i in range(len(data)):
            prediction = cls.predict_label(tree, data.iloc[i])
            if prediction == data.iloc[i][len(data.columns) - 1]:
                correct_predictions += weights[i]

        error = 1 - correct_predictions
        return error