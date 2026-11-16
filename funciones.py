import numpy as np

def gradient_descent(grad_fn, x0, learning_rate=0.1, n_iterations=100):
    x = np.array(x0, dtype=float)
    for _ in range(n_iterations):
        x = x - learning_rate * grad_fn(x)
    return x

def train_test_split(X, y, test_size=0.25, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std

def k_fold_indices(n_samples, k=5, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, folds[i]))
    return splits

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn

def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_cost(self, X, y):
        eps = 1e-15
        p = np.clip(self.predict_proba(X), eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        errors = self.predict_proba(X) - y
        dw = (1 / n_samples) * X.T.dot(errors)
        db = (1 / n_samples) * np.sum(errors)
        return dw, db

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self


from collections import Counter


class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_one(self, x):
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        return Counter(nearest_labels.tolist()).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tree_ = None

    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        features = np.arange(n_features)
        if self.max_features is not None:
            features = self.rng.choice(n_features, size=min(self.max_features, n_features), replace=False)
        best_feature, best_threshold, best_gini = None, None, np.inf
        for feature in features:
            for threshold in np.unique(X[:, feature]):
                mask = X[:, feature] <= threshold
                if mask.all() or not mask.any():
                    continue
                gini = (mask.sum() * self._gini(y[mask]) + (~mask).sum() * self._gini(y[~mask])) / n_samples
                if gini < best_gini:
                    best_feature, best_threshold, best_gini = feature, threshold, gini
        return best_feature, best_threshold

    def _build(self, X, y, depth):
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}
        mask = X[:, feature] <= threshold
        return {'leaf': False, 'feature': feature, 'threshold': threshold,
                'left': self._build(X[mask], y[mask], depth + 1),
                'right': self._build(X[~mask], y[~mask], depth + 1)}

    def fit(self, X, y):
        self.tree_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        while not node['leaf']:
            node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        return node['class']

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

class RandomForestClassifier:
    def __init__(self, n_trees=25, max_depth=5, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees_ = []
        self.oob_indices_ = []
        self.n_classes_ = None

    def _bootstrap_sample(self, X, y, rng):
        n = len(X)
        indices = rng.integers(0, n, size=n)
        return X[indices], y[indices], indices

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.n_classes_ = int(y.max()) + 1
        max_features = max(1, int(np.sqrt(X.shape[1])))
        self.trees_ = []
        self.oob_indices_ = []
        for _ in range(self.n_trees):
            X_b, y_b, idx = self._bootstrap_sample(X, y, rng)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=max_features, rng=rng)
            tree.fit(X_b, y_b)
            self.trees_.append(tree)
            self.oob_indices_.append(np.setdiff1d(np.arange(len(X)), idx))
        return self

    def predict(self, X):
        all_votes = np.array([tree.predict(X) for tree in self.trees_])
        return np.array([np.bincount(col, minlength=self.n_classes_).argmax() for col in all_votes.T])

    def oob_score(self, X, y):
        votes = np.zeros((len(X), self.n_classes_), dtype=int)
        for tree, oob in zip(self.trees_, self.oob_indices_):
            if len(oob) == 0:
                continue
            preds = tree.predict(X[oob])
            votes[oob, preds] += 1
        covered = votes.sum(axis=1) > 0
        return np.mean(votes[covered].argmax(axis=1) == y[covered])


class LinearSVM:
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.weights = None
        self.bias = None

    def decision_function(self, X):
        return X.dot(self.weights) + self.bias

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

    def compute_cost(self, X, y):
        margins = y * self.decision_function(X)
        hinge = np.maximum(0, 1 - margins)
        return 0.5 * np.sum(self.weights ** 2) + self.C * np.mean(hinge)

    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        violators = (y * self.decision_function(X)) < 1
        slack_signal = y * violators
        dw = self.weights - (self.C / n_samples) * X.T.dot(slack_signal)
        db = -(self.C / n_samples) * np.sum(slack_signal)
        return dw, db

    def fit(self, X, y):
        y_signed = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.n_iterations):
            dw, db = self.compute_gradients(X, y_signed)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


class NeuralNetwork:
    def __init__(self, hidden_size=8, learning_rate=0.5, n_iterations=3000, seed=0):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.seed = seed
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.loss_history_ = []

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 1, size=(n_features, self.hidden_size)) * np.sqrt(1 / n_features)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0, 1, size=self.hidden_size) * np.sqrt(1 / self.hidden_size)
        self.b2 = 0.0

    def _forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        return A1, sigmoid(Z2)

    def predict_proba(self, X):
        return self._forward(X)[1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def _backward(self, X, y, A1, p):
        n = X.shape[0]
        dZ2 = (p - y) / n
        dW2 = A1.T.dot(dZ2)
        db2 = dZ2.sum()
        dA1 = np.outer(dZ2, self.W2)
        dZ1 = dA1 * (1 - A1 ** 2)
        dW1 = X.T.dot(dZ1)
        db1 = dZ1.sum(axis=0)
        return dW1, db1, dW2, db2

    def fit(self, X, y):
        self._init_params(X.shape[1])
        self.loss_history_ = []
        for _ in range(self.n_iterations):
            A1, p = self._forward(X)
            self.loss_history_.append(binary_cross_entropy(y, p))
            dW1, db1, dW2, db2 = self._backward(X, y, A1, p)
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
        return self


