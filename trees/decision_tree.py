import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector, dtype=float)
    target_vector = np.asarray(target_vector).astype(int)

    n = feature_vector.shape[0]

    sort_idx = np.argsort(feature_vector, kind="mergesort")
    sorted_features = feature_vector[sort_idx]
    sorted_targets = target_vector[sort_idx]

    diff_mask = sorted_features[:-1] != sorted_features[1:]

    if not np.any(diff_mask):
        return np.array([]), np.array([]), None, None

    all_thresholds = (sorted_features[:-1] + sorted_features[1:]) / 2.0
    left_sizes = np.arange(1, n)           
    right_sizes = n - left_sizes
    cumsum_targets = np.cumsum(sorted_targets)   
    left_class1 = cumsum_targets[:-1]        
    total_class1 = cumsum_targets[-1]
    right_class1 = total_class1 - left_class1
    p1_left = left_class1 / left_sizes
    p1_right = right_class1 / right_sizes

    H_left = 1.0 - p1_left ** 2 - (1.0 - p1_left) ** 2
    H_right = 1.0 - p1_right ** 2 - (1.0 - p1_right) ** 2

    ginis_all = -(left_sizes / n) * H_left - (right_sizes / n) * H_right

    thresholds = all_thresholds[diff_mask]
    ginis = ginis_all[diff_mask]
    best_idx = int(np.argmax(ginis))
    threshold_best = float(thresholds[best_idx])
    gini_best = float(ginis[best_idx])

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(
                    lambda x: x[0],
                    sorted(ratio.items(), key=lambda x: x[1])
                ))
                categories_map = dict(zip(sorted_categories, range(len(sorted_categories))))

                feature_vector = np.array(list(map(
                    lambda x: categories_map[x], sub_X[:, feature]
                )), dtype=float)
            else:
                raise ValueError

            if np.unique(feature_vector).shape[0] < 2:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue

            if self._min_samples_leaf is not None:
                sorted_fv = np.sort(feature_vector)
                left_sizes = np.searchsorted(sorted_fv, thresholds, side="left")
                right_sizes = len(feature_vector) - left_sizes
                valid = (left_sizes >= self._min_samples_leaf) & \
                        (right_sizes >= self._min_samples_leaf)
                if not np.any(valid):
                    continue
                valid_thresholds = thresholds[valid]
                valid_ginis = ginis[valid]
                best_idx = int(np.argmax(valid_ginis))
                threshold = float(valid_thresholds[best_idx])
                gini = float(valid_ginis[best_idx])

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(
                        lambda x: x[0],
                        filter(lambda x: x[1] < threshold, categories_map.items())
                    ))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)],
                       node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
