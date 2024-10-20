import numpy as np
import random
from collections import Counter


class ImpurityMeasureMetrics:
    @staticmethod
    def entropy(values: np.ndarray) -> float:
        counts = Counter(values)
        total = sum(counts.values())
        probabilities = [count/total for count in counts.values()]
        entropy_ = -sum(p*np.log2(p) for p in probabilities if p>0)
        return entropy_
    
    @staticmethod
    def majority_error(values: np.ndarray) -> float:
        counts = Counter(values)
        total = sum(counts.values())
        max_count = max(counts.values()) 
        majority_error_ = 1 - (max_count/total)
        return majority_error_
    
    @staticmethod
    def gini_index(values: np.ndarray)-> float:
        counts = Counter(values)
        total = sum(counts.values())
        probabilities = [count/total for count in counts.values()]
        gini_index_ = 1 - sum(p ** 2 for p in probabilities)
        return gini_index_
    
    
class ID3DecisonTree():
    def __init__(self, dataset: np.ndarray, label_index: int, max_depth: int, impurity_measure_metric: str) -> None:
        self.dataset = dataset
        self.label_index = label_index
        self.max_depth = max_depth
        self.impurity_measure_metric = impurity_measure_metric
    
    def calculate_gain(self, data: np.ndarray, attribute: list) -> float:
        impurity_function = {
            "entropy": ImpurityMeasureMetrics.entropy,
            "majority_error": ImpurityMeasureMetrics.majority_error,
            "gini_index": ImpurityMeasureMetrics.gini_index
        }.get(self.impurity_measure_metric)

        if impurity_function is None:
            raise ValueError("Use 'entropy', 'majority_error', or 'gini_index' ")

        before_split_impurity = impurity_function(data[attribute])

        categories, categories_counts = np.unique(data[self.label_index], return_counts=True)

        after_split_impurity = 0
        total_counts = np.sum(categories_counts)

        for i in range(len(categories)):
            subset = data[data[self.label_index] == categories[i]][attribute]
            impurity = impurity_function(subset)
            
            after_split_impurity += (categories_counts[i] / total_counts)* impurity

        gain = before_split_impurity - after_split_impurity
        return gain
    
    def get_best_attribute_to_split(self, data: np.ndarray, attributes: list) -> int:
        item_values = [self.calculate_gain(data, attribute) for attribute in attributes]
        best_attribute_index = np.argmax(item_values)
        best_attribute = attributes[best_attribute_index]

        return best_attribute

    def remove_attribute(self, attributes: list, feature_to_remove: int) -> list:
        remaining_attributes = []
        for feature in attributes:
            if feature != feature_to_remove:
                remaining_attributes.append(feature)
        
        return remaining_attributes

    def get_majority_label(self, set: np.ndarray):
        labels = set[self.label_index]
        categories, categories_counts = np.unique(labels, return_counts=True)
        max_index = np.argmax(categories_counts)
        return categories[max_index]

    def construct_tree(self, current_set: np.ndarray, attributes: list, current_depth: int = 0, size_attribute_subset = 2, tree: dict = None) -> dict:
        if len(np.unique(current_set[self.label_index])) <= 1:
            return np.unique(current_set[self.label_index])[0]
        
        elif len(attributes) == 0:
            return self.get_majority_label(current_set)
        
        elif current_depth >= self.max_depth:
            return self.get_majority_label(current_set)
        
        elif len(current_set) == 0:
            return self.get_majority_label(self.dataset)
        
        else:
            current_depth += 1

            random_attributes = random.sample(attributes, size_attribute_subset) 
            best_attribute_to_split = self.get_best_attribute_to_split(current_set, attributes=random_attributes)

            tree = {best_attribute_to_split: {}}

            remaining_attributes = self.remove_attribute(attributes, best_attribute_to_split)
        
            for value in np.unique(current_set[best_attribute_to_split]):
                sub_set = current_set[current_set[best_attribute_to_split] == value]
                subtree = self.construct_tree(sub_set, remaining_attributes, current_depth, size_attribute_subset, tree)
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
    def calculate_error(cls, tree, data):
        correct_predictions = 0
        for i in range(len(data)):
            prediction = cls.predict_label(tree, data.iloc[i])
            if prediction == data.iloc[i][len(data.columns) - 1]:
                correct_predictions += 1

        accuracy = correct_predictions / len(data)
        error = 1 - accuracy
        return error